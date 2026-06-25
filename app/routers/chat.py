"""Chat-completion endpoint: the budget-guardrail hot path."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas import ChatRequest, ChatResponse, CompletionStatus
from app.services.budget import budget_store
from app.services.gemini_client import GeminiError, GeminiUnavailable, gemini_client
from app.services.predictor import predict_output
from app.services.retrain import feedback_buffer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/completions", response_model=ChatResponse)
async def create_completion(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> ChatResponse:
    """Estimate cost, enforce the budget, and (if allowed) generate a completion.

    Pipeline: identity check -> input-token count -> MoPE prediction -> budget
    guardrail -> generation. On generation failure the request degrades to a
    prediction-only response instead of erroring.
    """
    start = time.time()
    logger.info("[request] user_id=%s | prompt=%r", request.user_id, request.prompt[:80])

    # 1. Identity & budget state.
    quota = budget_store.get(request.user_id)
    if quota is None:
        logger.warning("[auth] REJECTED — unknown user_id=%r", request.user_id)
        raise HTTPException(status_code=401, detail="Unauthorized client")

    # 2. Input-token count (real, required for the guardrail).
    try:
        input_tokens = await gemini_client.count_input_tokens(request.prompt)
    except GeminiError as exc:
        raise HTTPException(status_code=502, detail=f"Inference Engine Error: {exc}") from exc

    # 3. MoPE output prediction (local, negligible latency).
    prediction = predict_output(request.prompt)
    predicted_output = prediction.predicted_output
    estimated_total = input_tokens + predicted_output
    logger.info(
        "[mope] input_tokens=%d | predicted_output=%d | estimated_total=%d | model=%s",
        input_tokens,
        predicted_output,
        estimated_total,
        prediction.model_status,
    )

    # 4. Guardrail: block before the expensive call if the estimate exceeds budget.
    if estimated_total > quota.available:
        logger.warning(
            "[guardrail] REJECTED_BY_BUDGET | user_id=%s | estimated=%d | available=%d",
            request.user_id,
            estimated_total,
            quota.available,
        )
        return ChatResponse(
            status=CompletionStatus.REJECTED_BY_BUDGET,
            texto_generado=None,
            metadata={
                "error": "Safeguard triggered: estimated output exceeds remaining budget.",
                "estimado": estimated_total,
                "disponible": quota.available,
                "model_status": prediction.model_status,
            },
        )

    # 5a. Fallback flow: generation unavailable -> return the estimate, do not charge.
    try:
        result = await gemini_client.generate(request.prompt)
    except GeminiUnavailable as exc:
        latency_ms = round((time.time() - start) * 1000, 2)
        logger.error("[result] GEMINI_UNAVAILABLE | user_id=%s | %s", request.user_id, exc)
        return ChatResponse(
            status=CompletionStatus.ALLOWED_PREDICTION_ONLY,
            texto_generado=None,
            metadata={
                "warning": "GEMINI_UNAVAILABLE",
                "detail": "Upstream generation failed; returning MoPE estimate only.",
                "input_tokens": input_tokens,
                "predicted_output": predicted_output,
                "estimated_total": estimated_total,
                "model_status": prediction.model_status,
                "latencia_ms": latency_ms,
            },
        )

    # 5b. Success flow: charge the real cost and decouple feedback from the response.
    actual_output = result.actual_output
    prediction_error = actual_output - predicted_output
    remaining = budget_store.charge(request.user_id, input_tokens + actual_output)
    background_tasks.add_task(
        feedback_buffer.add, request.prompt, input_tokens, actual_output
    )

    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info(
        "[result] ALLOWED | user_id=%s | actual_output=%d | prediction_error=%+d | "
        "remaining=%d | latency=%sms",
        request.user_id,
        actual_output,
        prediction_error,
        remaining,
        latency_ms,
    )

    return ChatResponse(
        status=CompletionStatus.ALLOWED,
        texto_generado=result.text,
        metadata={
            "input_tokens": input_tokens,
            "predicted_output": predicted_output,
            "actual_output": actual_output,
            "prediction_error": prediction_error,
            "latencia_ms": latency_ms,
            "presupuesto_restante": remaining,
            "model_status": prediction.model_status,
        },
    )
