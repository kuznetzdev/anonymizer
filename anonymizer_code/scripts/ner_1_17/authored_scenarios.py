"""Locally authored Russian synthetic scenarios for the NER 1.17 corpus.

The text is written here as annotated source, then markers are removed while
Unicode code-point offsets are computed from the resulting exact surfaces.
No external corpus or private 1.16 held-out data is used.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TypedDict


ACTIVE_LABELS = frozenset({
    "APART", "COUNTRY", "DISEASE", "EMAIL", "HOUSE", "INN", "INSTITUTION", "KPP",
    "LOCALITY", "OGRN", "ORG", "PASSPORT", "PER", "PHONE", "REGION", "SNILS", "STREET",
})
PRIORITY_LABELS = frozenset({
    "APART", "COUNTRY", "DISEASE", "HOUSE", "INSTITUTION", "LOCALITY", "ORG", "PASSPORT",
    "PER", "REGION", "SNILS", "STREET",
})
SPLITS = ("train", "validation", "test", "challenge")
MAX_TEXT_CHARS = 20_000
_MARKED = re.compile(r"\{\{([A-Z]+)\|([^{}]+)\}\}")
_IDENTIFIER_EXPR = re.compile(r'" \+ _(snils|inn|ogrn)\("(\d+)"\) \+ "')


class Entity(TypedDict):
    start: int
    end: int
    label: str


class AuthoredRecord(TypedDict):
    text: str
    entities: list[Entity]
    group: str
    template: str


def _snils(base: str) -> str:
    total = sum(int(digit) * weight for digit, weight in zip(base, range(9, 0, -1)))
    check = "00" if total in (100, 101) else f"{total % 101:02d}"
    if int(check) == 100:
        check = "00"
    return f"{base}{check}"


def _inn(base: str) -> str:
    weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
    check = str(sum(int(digit) * weight for digit, weight in zip(base, weights)) % 11 % 10)
    return base + check


def _ogrn(base: str) -> str:
    return base + str(int(base) % 11 % 10)


def _annotate(group: str, template: str, source: str) -> AuthoredRecord:
    parts: list[str] = []
    entities: list[Entity] = []
    cursor = 0
    for match in _MARKED.finditer(source):
        label, surface = match.groups()
        if label not in ACTIVE_LABELS:
            raise ValueError(f"Unsupported authored label: {label}")
        parts.append(source[cursor:match.start()])
        start = sum(map(len, parts))
        parts.append(surface)
        end = start + len(surface)
        entities.append({"start": start, "end": end, "label": label})
        cursor = match.end()
    parts.append(source[cursor:])
    text = "".join(parts)
    if _MARKED.search(text) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError("Authored source contains an invalid marker or text length")
    for entity in entities:
        if text[entity["start"]:entity["end"]] == "":
            raise ValueError("Authored entity has an empty surface")
    return {"text": text, "entities": entities, "group": group, "template": template}


def _resolve_identifier_expressions(source: str) -> str:
    functions = {"snils": _snils, "inn": _inn, "ogrn": _ogrn}
    return _IDENTIFIER_EXPR.sub(lambda match: functions[match.group(1)](match.group(2)), source)


def build_records() -> dict[str, list[AuthoredRecord]]:
    """Return independently authored train, validation, test and challenge notes.

    Held-out samples are intentionally modest and do not satisfy per-label
    statistical support for evaluation; support_report() makes that explicit.
    """
    raw: dict[str, list[tuple[str, str, str]]] = {
        "train": [
            ("train-admission-v1", "train-admission-intake-v1", """Приёмное отделение, 12.03.2026
Пациент: {{PER|Марина Лебедева}}. В анамнезе описана {{DISEASE|болезнь Паркинсона}}; в семейной выписке отдельно упомянута {{DISEASE|болезнь Паркинсона}} у родственника.
На плановую консультацию направлена в {{INSTITUTION|Городскую клиническую больницу № 8}}. Реагенты поставляет {{ORG|ООО «МедСнаб Урал»}}.
Адрес для вызова бригады: {{COUNTRY|Россия}}, {{REGION|Свердловская область}}, {{LOCALITY|Екатеринбург}}, {{STREET|улица Липовая}}, дом {{HOUSE|17к2}}, квартира {{APART|48}}.
СНИЛС {{SNILS|" + _snils("112233445") + "}}, паспорт {{PASSPORT|6604 123456}}. Контакт: {{PHONE|+7 912 345-67-89}}, {{EMAIL|marina@example.invalid}}.
Поставщик: ИНН {{INN|" + _inn("667788990") + "}}, КПП {{KPP|667701001}}, ОГРН {{OGRN|" + _ogrn("102660000000") + "}}.
Отрицательный контекст: у пациента Паркинсоновский корпус здания, а слово «больница» встречается в названии раздела старого шаблона; это не дополнительное учреждение или диагноз."""),
            ("train-longitudinal-v1", "train-longitudinal-course-v1", """Продольная запись наблюдения
Январь: пациент {{PER|Олег Миронов}} отмечал усиление тремора. Врач обсудил {{DISEASE|синдром Гийена—Барре}} как исключённую гипотезу, затем зафиксировал отсутствие новых неврологических симптомов.
Февраль: пациент перенёс консультацию в {{INSTITUTION|Клинический центр восстановительной медицины}}. Компания {{ORG|АО «Северная лабораторная логистика»}} передала расходные материалы.
Адрес временного проживания: {{COUNTRY|Казахстан}}, {{REGION|Алтайский край}}, {{LOCALITY|Барнаул}}, {{STREET|проспект Строителей}}, дом {{HOUSE|102/3}}, квартира {{APART|7}}.
Связь через {{PHONE|8 (800) 123-45-67}} или {{EMAIL|case@example.invalid}}. Архивный документ: СНИЛС {{SNILS|" + _snils("987654321") + "}}, паспорт {{PASSPORT|4508 765432}}.
Лабораторный поставщик указан отдельно: ИНН {{INN|" + _inn("770123456") + "}}, КПП {{KPP|770101001}}, ОГРН {{OGRN|" + _ogrn("106770000000") + "}}.
В журнале наблюдения зафиксировано спокойное дыхание в покое и отсутствие боли в грудной клетке. Давление измеряли утром и вечером одним поверенным прибором.
При контрольном визите врач сверил переносимость назначений и попросил отмечать время приёма препаратов. Пациент сообщил, что режим сна постепенно стабилизировался.
Результаты общего анализа крови сопоставили с предыдущим исследованием; клинически значимой динамики не обнаружено. Повторное лабораторное исследование назначено через месяц.
Специалист по реабилитации рекомендовал щадящую ежедневную нагрузку с короткими прогулками. Самостоятельно менять схему лечения до следующей консультации не следует.
На следующем визите команда проверит дневник симптомов и при необходимости скорректирует график наблюдения. Дополнительных контактов и адресных сведений в этой записи нет.
Пациент получил памятку о признаках ухудшения состояния и порядке обращения в дежурную службу. Содержание памятки обсудили устно, вопросов не осталось.
В итоговой части врач отметил хорошую переносимость обследования и отсутствие нежелательных реакций. Контрольная дата внесена в расписание отделения."""),
            ("train-negative-v1", "train-negative-qc-note-v1", """Контроль качества серии
Проба с кодом 0000-0000-0000 соответствует лабораторному контролю, это не СНИЛС. Телефонный канал проверки отключён; адрес электронной почты не предоставлялся.
Термин «двигательный» присутствует в названии алгоритма фильтрации, клинического диагноза в карточке нет. В журнале указан городской корпус, однако пациента и медицинскую организацию эта запись не называет."""),
            ("train-neuro-followup-v2", "train-neuro-visit-followup-v2", """Осмотр невролога, повторный визит
Пациент {{PER|Вера Кузнецова}} принесла дневник наблюдений. В выписке отмечена {{DISEASE|болезнь Гентингтона}}; фамилия Гентингтон в этой строке относится к названию заболевания.
Для сверки страховой записи предъявлен СНИЛС {{SNILS|" + _snils("563728194") + "}} и паспорт {{PASSPORT|4507 238641}}. Врач оставил копии документов в закрытой части медицинской карты."""),
            ("train-ocr-referral-v2", "train-ocr-scanned-referral-v2", """Скан направления, распознанный текст
Врачебная комиссия просит принять {{PER|Аркадия Тихонова}} в {{INSTITUTION|Городскую клиническую больницу № 12}}.
OCR-строка: «просп. Победы, д. 31А, кв. 8»; исходная адресная запись: {{COUNTRY|Россия}}, {{REGION|Воронежская область}}, {{LOCALITY|Воронеж}}, {{STREET|проспект Победы}}, дом {{HOUSE|31А}}, квартира {{APART|8}}.
СНИЛС {{SNILS|" + _snils("409876123") + "}}. На скане виден дефект печати в заголовке, отмеченный редактором как [неразборчиво]; пометка не меняет границы сущностей."""),
            ("train-inpatient-intake-v2", "train-inpatient-admission-record-v2", """Приёмный покой: регистрационная запись
{{PER|Денис Чернов}} доставлен на обследование по поводу {{DISEASE|синдрома Лайелла}}. Диагноз указан в направлении, подтверждение ожидается.
Приём оформлен в {{INSTITUTION|Центре экстренной медицинской помощи № 5}}. Анализатор передан по гарантии компанией {{ORG|ООО «УралМедТехника»}}.
Документы: паспорт {{PASSPORT|7206 419805}}, СНИЛС {{SNILS|" + _snils("753190864") + "}}. Номера относятся к одной карточке пациента."""),
            ("train-home-visit-v2", "train-home-care-address-v2", """Домашний патронаж, маршрут для медсестры
Визит к {{PER|Лидии Романовой}} назначен после консультации по поводу {{DISEASE|синдрома Рейно}}.
Адрес пациента: {{COUNTRY|Россия}}, {{REGION|Тульская область}}, {{LOCALITY|Новомосковск}}, {{STREET|улица Центральная}}, дом {{HOUSE|22}}, квартира {{APART|54}}.
Паспорт {{PASSPORT|7003 118209}} проверен при согласовании визита. В подъездной заметке просьба позвонить дежурному консьержу; личный телефон в бумажной копии не указан."""),
            ("train-discharge-admin-v2", "train-discharge-coordination-v2", """Координация выписки, административная часть
{{PER|Алиса Белова}} завершила курс лечения {{DISEASE|болезни Крона}} и ожидает контрольного приёма.
Медицинские документы направлены в {{INSTITUTION|Городской консультативно-диагностический центр № 6}}. Расходные материалы для перевязочной поставила {{ORG|АО «Медицинский склад Поволжье»}}.
Адрес для доставки выписки: {{COUNTRY|Россия}}, {{REGION|Самарская область}}, {{LOCALITY|Сызрань}}, {{STREET|улица Мира}}, дом {{HOUSE|9}}, квартира {{APART|12}}.
СНИЛС {{SNILS|" + _snils("821406357") + "}}, паспорт {{PASSPORT|3601 804217}}. Эти сведения внесены в реквизиты получателя, а название поставщика находится в отдельной строке."""),
            ("train-specialist-note-v2", "train-specialist-consult-note-v2", """Заключение консультанта
{{PER|Руслан Ефимов}} направлен на уточнение ранее установленной {{DISEASE|миастении гравис}}. Эпоним «гравис» является частью медицинского термина, отдельного пациента с таким именем в заключении нет.
Консультация проведена в {{INSTITUTION|Областной клинической больнице восстановительного лечения}}. Представитель поставщика {{ORG|ООО «Диагностические системы Балтика»}} оставил паспорт оборудования, который не относится к документам пациента."""),
            ("train-id-form-v2", "train-identity-form-v2", """Сверка регистрационной формы
{{PER|Зоя Громова}} предъявила паспорт {{PASSPORT|4502 671934}}. В поле СНИЛС указан номер {{SNILS|" + _snils("194857263") + "}}.
Адресные компоненты формы: {{REGION|Калужская область}}, {{LOCALITY|Обнинск}}, {{STREET|улица Курчатова}}, дом {{HOUSE|5}}, квартира {{APART|31}}. Страна гражданства — {{COUNTRY|Россия}}.
Оператор исправил перенос строки в OCR-копии: поле «кв.» не входит в значение номера квартиры."""),
            ("train-facility-supplier-v2", "train-facility-supplier-record-v2", """Журнал обслуживания оборудования
Выезд инженера согласован с {{INSTITUTION|Детской областной клинической больницей № 2}}. Поставку датчиков выполнила {{ORG|АО «Сигнал-Мед»}}.
В акте проверки приведена фамилия {{PER|Николая Орлова}}, ответственного за приёмку. Договор также содержит пункт о подозрении на {{DISEASE|синдром Чарга—Стросса}}, которое в карте пациента не подтверждено.
Технический серийный номер 0000-1234 не является паспортом или СНИЛС; он относится к прибору."""),
            ("train-negative-ocr-v2", "train-ocr-quality-control-v2", """Проверка качества OCR, без персональных сущностей
Система распознала «СНИЛС: 000 000 000 00», но поле заполнено демонстрационным нулевым значением. Серия паспорта оборудования — 1234 567890; к человеку номер не относится.
В шаблоне встретились слова «улица», «дом» и «клинический центр», однако адрес и собственное название учреждения отсутствуют. Текст передан на ручную проверку из-за смещения колонок."""),
            ("train-negative-medical-v2", "train-medical-index-negative-v2", """Фрагмент справочника для проверки поиска
Термины медицинского указателя перечислены по алфавиту, без связи с историей конкретного человека. Код поисковой задачи 123456789012 относится к индексу, не к документу.
В строке указано «ул. Центральная, дом 0» как пример допустимого поля. Это тестовая подпись формы, ни адреса пациента, ни названия организации в записи нет."""),
        ],
        "validation": [
            ("validation-neurology-v1", "validation-neurology-followup-v1", """Неврологический осмотр
Пациент {{PER|Екатерина Андреева}} наблюдается по поводу {{DISEASE|болезни Бехтерева}}; в семейном анамнезе отдельно упомянута болезнь Паркинсона у деда.
На контроль направлена в {{INSTITUTION|Детскую городскую поликлинику № 4}} вместе с представителем {{ORG|АО «ЛабТех-Сервис»}} для проверки оборудования.
Адрес регистрации: {{COUNTRY|Армения}}, {{REGION|Республика Татарстан}}, {{LOCALITY|Казань}}, {{STREET|Сосновый проезд}}, дом {{HOUSE|6Б}}, квартира {{APART|113}}.
СНИЛС {{SNILS|" + _snils("123456789") + "}}, паспорт {{PASSPORT|9201 654321}}. Контакты: {{PHONE|+7 (921) 234-56-78}}, {{EMAIL|followup@example.invalid}}.
Реквизиты учреждения-поставщика: ИНН {{INN|" + _inn("165432109") + "}}, КПП {{KPP|165401001}}, ОГРН {{OGRN|" + _ogrn("108160000000") + "}}."""),
            ("validation-clinical-negative-v1", "validation-lab-negative-v1", """Лабораторная отметка
Контрольный образец серии 1111-2222-3333 не является страховым номером. В заключении приведён пример поискового запроса без клинического диагноза и без связи с карточкой человека.
Значение «улица Сосновая» взято из учебного поля формы, фактический адрес не указан. Служебная фраза «городская поликлиника» не содержит названия конкретной организации."""),
        ],
        "test": [
            ("test-call-transcript-v2", "test-protected-handoff-transcript-v2", """Устная передача по защищённому каналу
Координатор: называю код карты; пациент {{PER|Илья Воронцов}} уже подтвердил данные.
Врач: в выписке указано {{DISEASE|синдром Гийена—Барре}}, направление адресовано в {{INSTITUTION|Минскую городскую клиническую больницу № 3}}.
Координатор: сопровождение обеспечит {{ORG|ООО «СеверМедПоставка»}}. Получатель живёт в {{COUNTRY|Беларусь}}, {{REGION|Минская область}}, {{LOCALITY|Минск}}, {{STREET|улица Машерова}}, дом {{HOUSE|18}}, квартира {{APART|42}}.
Врач: документ сверили со слов пациента: паспорт {{PASSPORT|5109 112233}}, СНИЛС {{SNILS|" + _snils("314159265") + "}}. Для обратного звонка записаны {{PHONE|+375 29 555-14-22}} и {{EMAIL|handoff@example.invalid}}.
Координатор: бухгалтерский реквизит поставщика — ИНН {{INN|" + _inn("780987654") + "}}, КПП {{KPP|780901001}}, ОГРН {{OGRN|" + _ogrn("110780000000") + "}}. Запись закончена."""),
            ("test-field-ledger-v2", "test-ocr-field-reconciliation-v2", """Лист сверки полей — корректура OCR
Персональная карточка = {{PER|Наталья Фёдорова}}; примечание специалиста: {{DISEASE|синдром Шегрена}}.
Направляющая организация = {{INSTITUTION|Республиканский центр восстановительной терапии}}; перевозчик оборудования = {{ORG|ООО «Балтика МедТранс»}}.
Гражданство = {{COUNTRY|Россия}} / субъект = {{REGION|Псковская область}} / город = {{LOCALITY|Псков}}.
Улица = {{STREET|улица Зелёная Роща}} / строение = дом {{HOUSE|14}} / помещение = квартира {{APART|29}}.
Документы после проверки цифр: паспорт {{PASSPORT|6002 998877}}, СНИЛС {{SNILS|" + _snils("271828182") + "}}.
Контакт для доставки: {{PHONE|+7 921 555-66-77}}, {{EMAIL|visit@example.invalid}}.
Контрагент, отдельная строка: ИНН {{INN|" + _inn("781234567") + "}}, КПП {{KPP|781201001}}, ОГРН {{OGRN|" + _ogrn("112780000000") + "}}.
Перед выдачей документ сверен по оригиналу; символы «8» и «В» в OCR-заголовке исправлены вне размеченных полей."""),
            ("test-negative-v1", "test-lab-protocol-negative-v1", """Лабораторный протокол
Партия реагента 123456789012 прошла внутреннюю проверку. Это код материала, а не паспорт или страховой номер.
Реестр содержит только технические статусы: повторная проверка завершена, отклонений нет. Именованных людей, адресов и клинических сведений в этой записи нет."""),
        ],
        "challenge": [
            ("challenge-review-board-v2", "challenge-discrepancy-board-minutes-v2", """Протокол комиссии по расхождениям в документах
Председатель открыла заседание по синтетическому делу {{PER|Софьи Крыловой}}. В приложенном заключении {{DISEASE|синдром Марфана}} повторно приведён как {{DISEASE|синдром Марфана}}; оба упоминания относятся к одному диагнозу в обсуждаемой карте.
Секретарь зафиксировала направление в {{INSTITUTION|Республиканский клинический центр № 2}}. Представитель поставщика {{ORG|АО «Вектор МедСервис»}} пояснил происхождение серийного номера диагностического прибора.
Поля адресной карточки проверены отдельно: страна — {{COUNTRY|Россия}}, регион — {{REGION|Республика Татарстан}}, населённый пункт — {{LOCALITY|Казань}}, улица — {{STREET|улица Академическая}}, дом — {{HOUSE|14Б}}, квартира — {{APART|26}}.
Документы, предъявленные при регистрации: паспорт {{PASSPORT|8603 445566}}, СНИЛС {{SNILS|" + _snils("864209753") + "}}. Для уведомления указаны {{PHONE|+7 921 444-55-66}} и {{EMAIL|board-case@example.invalid}}.
В закупочном приложении поставщика приведены ИНН {{INN|" + _inn("781112223") + "}}, КПП {{KPP|781101001}} и ОГРН {{OGRN|" + _ogrn("114780000000") + "}}. Комиссия подтвердила, что эти реквизиты относятся к юридическому лицу, а не к карте пациента.
Решение по пунктам: копии документов сверены с оригиналами; исправления OCR ограничить полями, где символы были повреждены; историю изменений сохранить в протоколе.
Вопрос о диагнозе направить лечащей команде для клинической интерпретации. Секретарь отметила, что одноимённые фамилии в справочниках не используются как дополнительное свидетельство о личности.
Закрытая часть совещания завершилась перечнем ответственных подразделений. Комиссия не меняла клинические назначения и не добавляла новых персональных реквизитов.
Итоговый лист включает дату сверки, источник каждого исправленного поля и подпись ответственного специалиста. Повторное заседание назначат только при поступлении нового документа."""),
            ("challenge-negative-v2", "challenge-index-integrity-negative-v2", """Сводка технического контроля индекса
Пакет с кодом 123456789012 прошёл проверку целостности; код является номером сборки программного индекса.
Файл содержит общие служебные заголовки «страна», «улица», «дом» и «электронная почта», но соответствующие значения пусты. Именованных лиц, адресов, организаций и записей о состоянии здоровья в пакете нет.
Нулевой маркер СНИЛС 0000 0000 0000 включён как демонстрационное значение поля. Он не связан с человеком; медицинские сведения в этом техническом фрагменте отсутствуют."""),
        ],
    }
    result = {
        split: [
            _annotate(group, template, _resolve_identifier_expressions(source))
            for group, template, source in raw[split]
        ]
        for split in SPLITS
    }
    validate_records(result)
    return result


def validate_records(records: dict[str, list[AuthoredRecord]]) -> None:
    if set(records) != set(SPLITS):
        raise ValueError("Authored records must provide exactly the declared splits")
    groups: dict[str, str] = {}
    templates: dict[str, str] = {}
    texts: set[str] = set()
    for split, rows in records.items():
        for row in rows:
            if not row["group"] or not row["template"] or row["text"] in texts:
                raise ValueError("Source IDs must exist and authored text must be unique")
            texts.add(row["text"])
            for registry, value in ((groups, row["group"]), (templates, row["template"])):
                previous = registry.setdefault(value, split)
                if previous != split:
                    raise ValueError(f"Source group or template crosses splits: {value}")
            if not row["text"].strip() or len(row["text"]) > MAX_TEXT_CHARS:
                raise ValueError("Authored text is empty or exceeds the importer limit")
            previous_end = 0
            for entity in row["entities"]:
                start, end, label = entity["start"], entity["end"], entity["label"]
                if label not in ACTIVE_LABELS or start < previous_end or end <= start or end > len(row["text"]):
                    raise ValueError("Authored entity span is invalid, overlapping, or unsupported")
                if not row["text"][start:end].strip():
                    raise ValueError("Authored entity surface is empty")
                if label in {"SNILS", "INN", "OGRN"} and not _valid_checksum(label, row["text"][start:end]):
                    raise ValueError(f"Invalid {label} checksum")
                previous_end = end
    for split in SPLITS:
        if not records[split]:
            raise ValueError(f"Split {split} has no authored records")


def _valid_checksum(label: str, value: str) -> bool:
    digits = "".join(char for char in value if char.isdigit())
    if label == "SNILS":
        return len(digits) == 11 and _snils(digits[:9]) == digits
    if label == "INN":
        return len(digits) == 10 and _inn(digits[:9]) == digits
    if label == "OGRN":
        return len(digits) == 13 and _ogrn(digits[:12]) == digits
    return True


def support_report(records: dict[str, list[AuthoredRecord]], minimum: int = 30) -> dict[str, dict[str, object]]:
    """Report sparse split support honestly; this is not an evaluation gate."""
    if minimum < 1:
        raise ValueError("Minimum support must be positive")
    report: dict[str, dict[str, object]] = {}
    for split in ("test", "challenge"):
        counts = Counter(
            entity["label"] for row in records[split] for entity in row["entities"]
        )
        report[split] = {
            "support": {label: counts[label] for label in sorted(ACTIVE_LABELS)},
            "blocked_labels": sorted(label for label in PRIORITY_LABELS if counts[label] < minimum),
            "evaluation_ready": all(counts[label] >= minimum for label in PRIORITY_LABELS),
        }
    return report
