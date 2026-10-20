"""Routing metadata for concrete NER labels; parents are not model labels."""

DOCUMENT_LABEL_DESCRIPTIONS = {
    "PASSPORT": "Серия и номер паспорта",
    "OMS_POLICY_ID": "Номер полиса обязательного медицинского страхования",
    "PATIENT_CARD_ID": "Номер карты пациента",
    "MEDICAL_RECORD_ID": "Номер медицинской карты или истории болезни",
    "REFERRAL_ID": "Номер направления или медицинского талона",
    "DRIVER_LICENSE_ID": "Номер водительского удостоверения",
    "OTHER_DOCUMENT_ID": "Номер документа явно указанного типа, не покрытого отдельной меткой",
}

DOCUMENT_PARENT_CLASS = "DOCUMENT"


def category_parent(label: str) -> str | None:
    return DOCUMENT_PARENT_CLASS if label in DOCUMENT_LABEL_DESCRIPTIONS else None
