"""Independently authored document-ID contrasts for the next corpus release.

All examples are fictional. DOCUMENT is routing metadata; spans carry one
concrete leaf label so the existing BIO classifier remains flat.
"""

from __future__ import annotations

import re
from typing import TypedDict


DOCUMENT_PARENT = "DOCUMENT"
DOCUMENT_LEAF_LABELS = frozenset({
    "PASSPORT",
    "OMS_POLICY_ID",
    "PATIENT_CARD_ID",
    "MEDICAL_RECORD_ID",
    "REFERRAL_ID",
    "DRIVER_LICENSE_ID",
    "OTHER_DOCUMENT_ID",
})
PARENT_BY_LABEL = {label: DOCUMENT_PARENT for label in DOCUMENT_LEAF_LABELS}
_LABELS = DOCUMENT_LEAF_LABELS | {
    "APART", "DISEASE", "HOUSE", "INSTITUTION", "LOCALITY", "ORG", "PER",
    "PHONE", "REGION", "SNILS", "STREET",
}
_SPLITS = ("train", "validation", "test", "challenge")
_MARKER = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\|([^\[\]]+)\]\]")


class Entity(TypedDict):
    start: int
    end: int
    label: str


class Record(TypedDict):
    text: str
    entities: list[Entity]
    group: str
    template: str


def _record(group: str, template: str, marked_text: str) -> Record:
    """Remove inline markers and calculate Python code-point offsets."""
    parts: list[str] = []
    entities: list[Entity] = []
    cursor = 0
    length = 0
    for match in _MARKER.finditer(marked_text):
        label, surface = match.groups()
        if label not in _LABELS:
            raise ValueError(f"Unsupported label: {label}")
        prefix = marked_text[cursor:match.start()]
        parts.append(prefix)
        length += len(prefix)
        start = length
        parts.append(surface)
        length += len(surface)
        entities.append({"start": start, "end": length, "label": label})
        cursor = match.end()
    parts.append(marked_text[cursor:])
    text = "".join(parts)
    if not text.strip() or _MARKER.search(text) or "[[" in text or "]]" in text:
        raise ValueError("Malformed marked record")
    if any(text[e["start"]:e["end"]] == "" for e in entities):
        raise ValueError("Empty entity span")
    return {"text": text, "entities": entities, "group": group, "template": template}


def build_records() -> dict[str, list[Record]]:
    """Build fresh, split-isolated mixed notes; heldouts are not corpus-derived."""
    raw: dict[str, list[tuple[str, str, str]]] = {
        "train": [
            ("doc-train-intake-a", "doc-template-intake-a", """Дежурная запись: [[PER|Ирина Сафонова]] доставлена в [[INSTITUTION|Клинический пункт Северный]] после приступа, похожего на [[DISEASE|болезнь Бехтерева]]. На карте улиц указан адрес: [[LOCALITY|Киров]], [[STREET|улица Академика Королёва]], дом [[HOUSE|26]], квартира [[APART|14]]. Королёв в строке адреса — часть названия улицы; [[PER|Сергей Королёв]] упомянут отдельно как автор старого заключения. В регистратуре сверили паспорт [[PASSPORT|71 08 483921]], полис ОМС [[OMS_POLICY_ID|ОМС-64-39017528]], карту пациента [[PATIENT_CARD_ID|КП-2026-00418]] и СНИЛС [[SNILS|112-233-445 95]]. Для звонка оставлен номер [[PHONE|+7 921 406-28-53]]. Кабинет 214, давление 118/76, доза 5 мг, дата визита 14.02.2026 и код пробирки 483921 — служебные или медицинские числа, не реквизиты документа."""),
            ("doc-train-discharge-b", "doc-template-discharge-b", """В выписке [[PER|Михаил Горин]] описана ремиссия после лечения [[DISEASE|болезни Ходжкина]]; фраза «исключить болезнь Грейвса» относится к дифференциальному поиску, обе болезни остаются медицинскими упоминаниями. Пациента наблюдали в [[INSTITUTION|Городском центре восстановительной терапии]], расходные материалы поставило [[ORG|ООО «Тихий берег Мед»]]. Архивная история болезни имеет номер [[MEDICAL_RECORD_ID|ИБ-26-731590]], направление на контроль — [[REFERRAL_ID|НП-8041-2602]]. Врачебный кабинет 731590 и номер лабораторной пробы 80412602 не являются документами. Для связи: [[PHONE|8 (812) 604-19-37]]. В семейном анамнезе отмечена [[DISEASE|болезнь Паркинсона]] у матери; инженер [[PER|Алексей Паркинсон]] упомянут в приложенном акте о ремонте аппарата. На запасном бланке водительское удостоверение записано как [[DRIVER_LICENSE_ID|ВУ 77 16 584203]], а паспорт — [[PASSPORT|45 12 908314]]."""),
            ("doc-train-home-c", "doc-template-home-c", """Медсестра посетила [[PER|Надежду Волкову]] по адресу: [[LOCALITY|Тула]], [[STREET|улица Пирогова]], дом [[HOUSE|8Б]], квартира [[APART|63]]. Имя [[PER|Николай Пирогов]] стоит в отдельной строке как автор исторической справки, оно не часть адреса. Врач уточнял симптомы [[DISEASE|синдрома Марфана]] и отсутствие признаков [[DISEASE|болезни Альцгеймера]]. Предъявлены полис [[OMS_POLICY_ID|ОМС-71-56018493]], карточка [[PATIENT_CARD_ID|КП-118-0526]], медкарта [[MEDICAL_RECORD_ID|МК-2026-145802]] и направление [[REFERRAL_ID|НП-19-670245]]. Числа 145802, 19.06.2026, 36,7 °C, 120/80 и 500 мг относятся к пробе, дате, температуре, давлению и дозировке. Неизвестный номер на вложенном листе явно назван номером документа: [[OTHER_DOCUMENT_ID|ДК-2026-009731]]."""),
            ("doc-train-admin-d", "doc-template-admin-d", """В отделении [[INSTITUTION|Областная клиника дневного наблюдения]] получатель [[PER|Алёна Белова]] подписала согласие. Болезнь [[DISEASE|Шарко—Мари—Тута]] указана как рабочая гипотеза; в соседнем абзаце врач [[PER|Жан-Мартен Шарко]] цитируется в исторической справке. Адрес филиала: [[REGION|Пермский край]], [[LOCALITY|Пермь]], [[STREET|улица Менделеева]], дом [[HOUSE|41]], квартира [[APART|9]]. Номер направления [[REFERRAL_ID|НП-5902-771046]] относится к пациенту, а запись «комната 5902» — к расписанию. На обороте есть водительское удостоверение [[DRIVER_LICENSE_ID|ВУ 59 18 640275]], паспорт [[PASSPORT|59 04 761829]] и СНИЛС [[SNILS|987-654-321 83]]. Технический код аппарата 640275 и доза 0,25 мг не должны приниматься за идентификаторы."""),
        ],
        "validation": [
            ("doc-validation-neuro-e", "doc-template-validation-neuro-e", """Контрольный осмотр [[PER|Оксаны Мироновой]] прошёл в [[INSTITUTION|Медицинском кабинете Приозёрный]]. Врач исключает [[DISEASE|болезнь Кушинга]], но подтверждает ранее описанную [[DISEASE|болезнь Крона]]. В пояснении к старой публикации назван [[PER|Харви Кушинг]], пациентом он не является. Страховой полис [[OMS_POLICY_ID|ОМС-52-00619384]] сверили с паспортом [[PASSPORT|52 11 264908]] и картой [[PATIENT_CARD_ID|КП-052-4419]]. На бланке указан телефон [[PHONE|+7 905 731-40-62]]. Номер кабинета 4419, рост 164 см, дата 02.07.2026 и лабораторный показатель 52 не являются документными номерами."""),
            ("doc-validation-referral-f", "doc-template-validation-referral-f", """В журнале вызова записаны [[PER|Роман Елисеев]], адрес [[LOCALITY|Рязань]], [[STREET|проспект Гагарина]], дом [[HOUSE|17]], квартира [[APART|5]] и телефон [[PHONE|+7 920 531-08-46]]. Гагарин в адресе — только часть названия проспекта; [[PER|Юрий Гагарин]] отдельно упомянут в описании экспоната. Врач обсуждает [[DISEASE|синдром Туретта]] как неподтверждённую версию и отмечает, что слово «Туретт» относится к эпониму. Направление [[REFERRAL_ID|НП-62-190437]] и медкарта [[MEDICAL_RECORD_ID|МК-26-007614]] принадлежат пациенту; паспорт прибора [[OTHER_DOCUMENT_ID|ДП-80-315790]] приложен к сервисному акту. Число 190437 также встречается в поле количества импульсов аппарата — это значение измерения, не отдельный документ."""),
            ("doc-validation-mixed-k", "doc-template-validation-mixed-k", """После консультации в [[INSTITUTION|Городском диагностическом отделении Южное]] пациент [[PER|Дарья Мельникова]] получила план наблюдения при [[DISEASE|болезни Крона]]. В справочной цитате назван врач [[PER|Бёррилл Крон]], а строка адреса содержит [[STREET|улица Сеченова]]; [[PER|Иван Сеченов]] упомянут в пояснении к названию улицы. Адрес: [[LOCALITY|Липецк]], дом [[HOUSE|19]], квартира [[APART|27]]. Сотрудник сверил паспорт [[PASSPORT|48 03 275914]], полис [[OMS_POLICY_ID|ОМС-48-19370562]], карту [[PATIENT_CARD_ID|КП-48-031762]], историю болезни [[MEDICAL_RECORD_ID|ИБ-48-052931]], направление [[REFERRAL_ID|НП-48-604218]], водительское удостоверение [[DRIVER_LICENSE_ID|ВУ 48 12 730416]] и иной документ с явной пометкой «номер документа» [[OTHER_DOCUMENT_ID|ДК-48-009264]]. СНИЛС [[SNILS|314-159-265 52]] внесён в отдельное поле, телефон для обратного звонка — [[PHONE|+7 910 284-63-57]]. Кабинет 27, дата 21.07.2026, проба 604218, пульс 68 и доза 20 мг служат отрицательными числовыми примерами."""),
        ],
        "test": [
            ("doc-test-mixed-g", "doc-template-test-mixed-g", """При переводе [[PER|Ларисы Климовой]] из [[INSTITUTION|Центра неотложной помощи Заречье]] в заключении перечислены [[DISEASE|болезнь Бехтерева]] и подозрение на [[DISEASE|синдром Гийена—Барре]]. Историческую заметку подписал [[PER|Жорж Гийен]]; фамилия в названии улицы [[STREET|улица Гийена]] обозначает топоним. Адрес: [[LOCALITY|Ижевск]], дом [[HOUSE|30]], квартира [[APART|22]]. Полис [[OMS_POLICY_ID|ОМС-18-40627195]], водительское удостоверение [[DRIVER_LICENSE_ID|ВУ 18 14 907263]], паспорт [[PASSPORT|18 09 340716]] и произвольная справка [[OTHER_DOCUMENT_ID|СД-18-005842]] перечислены отдельно. Номер койки 22, проба 907263, пульс 72 и дата 11.08.2026 — отрицательные числовые контексты."""),
            ("doc-test-record-h", "doc-template-test-record-h", """В карте [[PER|Виктора Смирнова]] специалист [[PER|Иван Павлов]] описал [[DISEASE|синдром Рейтера]] и отдельно упомянул [[DISEASE|болезнь Фабри]]. Для повторного приёма выдан полис [[OMS_POLICY_ID|ОМС-03-72845160]], карточка [[PATIENT_CARD_ID|КП-73-9016]], история [[MEDICAL_RECORD_ID|ИБ-73-450281]] и талон направления [[REFERRAL_ID|НП-73-662094]]. Название [[STREET|улица Павлова]] указано как место регистрации, а врач Павлов назван человеком в отдельной фразе. На соседнем листе номер 450281 обозначает пробирку; 662094 — код аппарата. Температура 37,1 °C, дата 19.09.2026, кабинет 16 и доза 2 мг остаются обычными числами."""),
            ("doc-test-admission-l", "doc-template-test-admission-l", """Приём пациентки [[PER|Полины Яровой]] в [[INSTITUTION|Межрайонном медицинском центре Роща]] оформлен после обострения [[DISEASE|болезни Боткина]]. В старом справочнике [[STREET|улица Боткина]] указана как адрес филиала; [[PER|Сергей Боткин]] назван в исторической справке отдельно. Адрес проживания: [[LOCALITY|Курск]], дом [[HOUSE|3]], квартира [[APART|51]]. В форме есть паспорт [[PASSPORT|46 07 351820]], полис [[OMS_POLICY_ID|ОМС-46-08521947]], карта [[PATIENT_CARD_ID|КП-46-050183]], история [[MEDICAL_RECORD_ID|ИБ-46-210704]], направление [[REFERRAL_ID|НП-46-820531]], водительское удостоверение [[DRIVER_LICENSE_ID|ВУ 46 16 304792]] и другой документ, прямо отмеченный как удостоверение личности, [[OTHER_DOCUMENT_ID|УД-46-062418]]. СНИЛС [[SNILS|271-828-182 91]] проверен в отдельном поле, контактный телефон [[PHONE|+7 915 327-48-60]]. Проба 820531, кабинет 51, температура 37,4 °C, дата 22.08.2026 и доза 250 мг не являются документами."""),
        ],
        "challenge": [
            ("doc-challenge-eponym-i", "doc-template-challenge-eponym-i", """Пациент [[PER|Екатерина Руднева]] проходит обследование в [[INSTITUTION|Клинике семейного здоровья Восток]]. В анамнезе указана [[DISEASE|болезнь Фон Реклингхаузена]]; врач отдельно проверяет отсутствие [[DISEASE|болезни Бехтерева]]. В адресе стоит [[STREET|улица Руднева]], дом [[HOUSE|6]], а [[PER|Пётр Руднев]] встречается в списке дежурных сотрудников. На стойке регистратуры сверили паспорт [[PASSPORT|40 05 812639]], полис [[OMS_POLICY_ID|ОМС-40-81730526]], номер истории [[MEDICAL_RECORD_ID|ИБ-40-120693]] и СНИЛС [[SNILS|563-728-194 21]]. Направление [[REFERRAL_ID|НП-40-007251]] лежит рядом с результатом пробы 007251; в этом втором употреблении число — лабораторный код. Дата 07.10.2026, давление 126/82 и доза 10 мг не обозначают документы."""),
            ("doc-challenge-admin-j", "doc-template-challenge-admin-j", """У [[PER|Станислава Орехова]] в записи приёмного отделения отмечены [[DISEASE|болезнь Грейвса]] и семейное упоминание [[DISEASE|синдрома Марфана]]. Справочник улиц содержит [[STREET|улицу Грейвса]], а в исследовательской цитате назван [[PER|Роберт Джеймс Грейвс]]. Пациент зарегистрирован в [[LOCALITY|Костроме]], дом [[HOUSE|12]], квартира [[APART|38]]. В комплекте: паспорт [[PASSPORT|44 13 927501]], полис [[OMS_POLICY_ID|ОМС-44-73195028]], карта [[PATIENT_CARD_ID|КП-44-003815]], история болезни [[MEDICAL_RECORD_ID|ИБ-44-480216]], направление [[REFERRAL_ID|НП-44-028417]], права [[DRIVER_LICENSE_ID|ВУ 44 09 615308]], СНИЛС [[SNILS|409-876-123 14]] и бумага с явным полем «номер документа» [[OTHER_DOCUMENT_ID|ДК-44-071583]]. В той же строке температура 38,2 °C, время 09:40 и проба 615308; эти числа не являются документами. Телефон для связи: [[PHONE|+7 903 842-17-56]]."""),
        ],
    }
    records = {
        split: [_record(group, template, text) for group, template, text in rows]
        for split, rows in raw.items()
    }
    validate_records(records)
    return records


def validate_records(records: dict[str, list[Record]]) -> None:
    if set(records) != set(_SPLITS):
        raise ValueError("Expected train, validation, test and challenge splits")
    seen_groups: set[str] = set()
    seen_templates: set[str] = set()
    for split in _SPLITS:
        for record in records[split]:
            if record["group"] in seen_groups or record["template"] in seen_templates:
                raise ValueError("Source groups and templates must be split-isolated")
            seen_groups.add(record["group"])
            seen_templates.add(record["template"])
            previous_end = 0
            for entity in record["entities"]:
                start, end, label = entity["start"], entity["end"], entity["label"]
                if label not in _LABELS or not 0 <= start < end <= len(record["text"]):
                    raise ValueError("Invalid entity label or span bounds")
                if start < previous_end:
                    raise ValueError("Entity spans must be ordered and non-overlapping")
                previous_end = end
