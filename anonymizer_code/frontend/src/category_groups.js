export const CATEGORY_PARENT_LABELS = { DOCUMENT: 'Документы' };

export function groupCategories(categories) {
  const groups = new Map();
  for (const category of categories) {
    const parentClass = category.parent_class ?? '';
    if (!groups.has(parentClass)) groups.set(parentClass, []);
    groups.get(parentClass).push(category);
  }
  return [...groups]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([parentClass, items]) => ({ parentClass, items }));
}
