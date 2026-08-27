"""Input form for the credit risk assessment interface."""
from django import forms

CLIENT_TYPE_CHOICES = [
    ("Rural", "Rural"),
    ("Semi-urban", "Semi-urban"),
    ("Urban", "Urban"),
]

REPAY_MODE_CHOICES = [
    ("N", "N — Standard"),
    ("I", "I — Instalment"),
]


class AssessmentForm(forms.Form):
    """Captures the borrower and facility attributes the model requires.

    Gender and marital status are deliberately absent. They were excluded from
    the model feature space on fair-lending grounds (Section 4.3 of the analysis
    notebook) and are therefore not collected here.
    """

    investment_total = forms.FloatField(
        label="Total facility amount",
        min_value=0,
        initial=1_500_000,
        help_text="Principal value of the loan facility",
        widget=forms.NumberInput(attrs={"step": "1000"}),
    )
    current_balance = forms.FloatField(
        label="Current outstanding balance",
        min_value=0,
        initial=25_000,
        help_text="Amount currently outstanding on the account",
        widget=forms.NumberInput(attrs={"step": "1000"}),
    )
    install_size = forms.FloatField(
        label="Instalment amount",
        min_value=0,
        initial=0,
        help_text="Scheduled repayment per instalment (enter 0 if none recorded)",
        widget=forms.NumberInput(attrs={"step": "100"}),
    )
    due_payment = forms.FloatField(
        label="Amount in arrears",
        min_value=0,
        initial=0,
        help_text="Overdue amount on the account (enter 0 if current)",
        widget=forms.NumberInput(attrs={"step": "100"}),
    )
    client_type = forms.ChoiceField(
        label="Client settlement type",
        choices=CLIENT_TYPE_CHOICES,
        initial="Rural",
    )
    repay_mode = forms.ChoiceField(
        label="Repayment mode",
        choices=REPAY_MODE_CHOICES,
        initial="N",
    )

    def clean(self):
        cleaned = super().clean()
        total = cleaned.get("investment_total")
        balance = cleaned.get("current_balance")
        if total is not None and balance is not None and total > 0 and balance > total * 5:
            self.add_error(
                "current_balance",
                "Outstanding balance is implausibly large relative to the facility.",
            )
        return cleaned
