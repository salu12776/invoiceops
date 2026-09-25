"""The shape every extracted invoice must have. The model is forced to answer in this schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="Item or service name as printed")
    quantity: float | None = Field(default=None, description="Quantity or hours")
    unit_price: float | None = Field(default=None, description="Price per unit")
    amount: float | None = Field(default=None, description="Line total as printed")


class InvoiceData(BaseModel):
    is_invoice: bool = Field(description="True only for an invoice, bill or receipt. False for anything else")
    vendor: str | None = Field(default=None, description="Company that issued the invoice (the seller)")
    invoice_number: str | None = Field(default=None, description="Invoice or receipt number exactly as printed")
    date: str | None = Field(default=None, description="Issue date as YYYY-MM-DD (not the due date)")
    currency: str | None = Field(default=None, description="ISO code such as PKR, USD, EUR")
    subtotal: float | None = Field(default=None, description="Sum of line items before discount and tax")
    discount: float | None = Field(default=None, description="Discount amount as a positive number, 0 if none")
    tax: float | None = Field(default=None, description="Tax / GST / VAT amount, 0 if none")
    total: float | None = Field(default=None, description="Final total or amount due, exactly as printed")
    line_items: list[LineItem] = Field(default_factory=list)
