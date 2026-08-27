from django.apps import AppConfig


class AssessmentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "assessment"

    def ready(self):
        # Warm the model on startup so the first request is not slow.
        from assessment import ml
        try:
            ml.get_engine()
        except Exception as exc:  # pragma: no cover
            print(f"[assessment] Model warm-up deferred: {exc}")
