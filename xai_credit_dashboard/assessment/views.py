"""Views for the credit risk assessment prototype."""
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from assessment import ml
from assessment.forms import AssessmentForm


def assess(request):
    """Main assessment page: form on GET, prediction plus explanation on POST."""
    result = None
    error = None
    form = AssessmentForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        try:
            engine = ml.get_engine()
            result = engine.assess(form.cleaned_data)
        except FileNotFoundError as exc:
            error = str(exc)
        except Exception as exc:  # pragma: no cover
            error = f"Assessment failed: {exc}"

    try:
        metadata = ml.get_engine().metadata
        global_importance = ml.get_engine().global_importance
    except Exception:
        metadata, global_importance = {}, []

    return render(request, "assessment/index.html", {
        "form": form,
        "result": result,
        "error": error,
        "metadata": metadata,
        "global_importance": global_importance,
    })


def about(request):
    """Methodology and model provenance page."""
    try:
        engine = ml.get_engine()
        metadata = engine.metadata
        global_importance = engine.global_importance
    except Exception:
        metadata, global_importance = {}, []
    return render(request, "assessment/about.html", {
        "metadata": metadata,
        "global_importance": global_importance,
    })


@csrf_exempt
def api_predict(request):
    """JSON endpoint. Accepts the same fields as the form."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        payload = json.loads(request.body.decode() or "{}")
        form = AssessmentForm(payload)
        if not form.is_valid():
            return JsonResponse({"error": "validation failed",
                                 "details": form.errors}, status=400)
        result = ml.get_engine().assess(form.cleaned_data)
        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
