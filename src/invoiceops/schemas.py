"""The shape every extracted invoice must have. The model is forced to answer in this schema."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    description: str = Field(description="Item or service name as printed")
    quantity: float | None = Field(default=None, description="Quantity or hours")
    unit_price: float | None = Field(default=None, description="Price per unit")
    amount: float | None = Field(default=None, description="Line total as printed")


class Charge(BaseModel):
    label: str = Field(description="Name of the charge as printed, e.g. Shipping, FPA, TV fee")
    amount: float = Field(description="Amount as printed, a positive number")


class InvoiceData(BaseModel):
    is_invoice: bool = Field(description="True only for an invoice, bill or receipt. False for anything else")
    vendor: str | None = Field(default=None, description="Company that issued the invoice (the seller)")
    invoice_number: str | None = Field(default=None, description="Invoice or receipt number exactly as printed")
    date: str | None = Field(default=None, description="Issue date as YYYY-MM-DD (not the due date)")
    currency: str | None = Field(default=None, description="ISO code such as PKR, USD, EUR")
    subtotal: float | None = Field(default=None, description="Sum of line items before discount and tax")
    discount: float | None = Field(default=None, description="Discount amount as a positive number, 0 if none")
    tax: float | None = Field(default=None, description="Tax / GST / VAT amount, 0 if none")
    other_charges: list[Charge] = Field(
        default_factory=list,
        description="Every other amount added on top of subtotal and tax (shipping, delivery, service charge, "
                    "fees, duties, surcharges), each listed separately. Do not add them up. Empty if none",
    )
    total: float | None = Field(default=None, description="Final total or amount due, exactly as printed")
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("other_charges", mode="before")
    @classmethod
    def _accept_number(cls, v):
        """Be forgiving: if a model still sends one number (or null), turn it into a list."""
        if v is None:
            return []
        if isinstance(v, (int, float)):
            return [{"label": "Other charges", "amount": v}] if v else []
        return v

    @property
    def other_total(self) -> float:
        """Python adds the charges up, not the model."""
        return round(sum(c.amount for c in self.other_charges), 2)