"""Routing metadata for concrete NER labels; parents are not model labels."""

DOCUMENT_LABELS = frozenset({
    "PASSPORT",
    "OMS_POLICY_ID",
    "MEDICAL_RECORD_ID",
    "REFERRAL_ID",
    "DRIVER_LICENSE_ID",
    "OTHER_DOCUMENT_ID",
})

# Fallback descriptions are used before ontology startup and for legacy rows.
DOCUMENT_LABEL_DESCRIPTIONS = {
    "PASSPORT": "Серия и номер паспорта",
    "OMS_POLICY_ID": "Номер полиса обязательного медицинского страхования",
    "PATIENT_CARD_ID": "Устаревший синоним MEDICAL_RECORD_ID: номер карты пациента.",
    "MEDICAL_RECORD_ID": "Номер карты пациента, амбулаторной или медицинской карты либо истории болезни; эти названия относятся к одному типу записи.",
    "REFERRAL_ID": "Номер оформленного направления пациента или выданного талона-направления; общий талон электронной очереди сам по себе не относится к этой категории.",
    "DRIVER_LICENSE_ID": "Серия или номер водительского удостоверения; название документа и слова-подсказки не включаются в span.",
    "OTHER_DOCUMENT_ID": "Номер явно названного документа или удостоверения без отдельной метки, включая паспорт оборудования; неизвестная цифровая строка без документного контекста сюда не относится.",
}

DOCUMENT_PARENT_CLASS = "DOCUMENT"


def category_parent(label: str) -> str | None:
    if label in DOCUMENT_LABELS or label in DOCUMENT_LABEL_DESCRIPTIONS:
        return DOCUMENT_PARENT_CLASS
    return None
