<template>
  <div class="selected-params">
    <div class="params-content">
      <div class="region-group">
        <div class="icon-wrapper">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 64 64"
            fill="currentColor"
            class="geo-icon"
          >
            <path
              fill-rule="evenodd"
              d="M32,0C18.746,0,8,10.746,8,24c0,5.219,1.711,10.008,4.555,13.93c0.051,0.094,0.059,0.199,0.117,0.289l16,24  C29.414,63.332,30.664,64,32,64s2.586-0.668,3.328-1.781l16-24c0.059-0.09,0.066-0.195,0.117-0.289C54.289,34.008,56,29.219,56,24  C56,10.746,45.254,0,32,0z M32,32c-4.418,0-8-3.582-8-8s3.582-8,8-8s8,3.582,8,8S36.418,32,32,32z"
              clip-rule="evenodd"
            />
          </svg>
        </div>
        <div class="region-info">
          <p class="label">Выбранный регион</p>
          <h2 class="region-name">{{ props.selectedPath || 'Российская Федерация' }}</h2>
        </div>
      </div>
      <div class="customer-info">
        <p class="label">Заказчик</p>
        <!-- <div class="customer-name-wrapper"> -->
          <p class="customer-name">
            {{ props.filters.customer || 'Все заказчики' }}
          </p>
        <!-- </div> -->
      </div>
      <div class="date-group">
        <p class="label">Период</p>
        <p class="date-value">{{ formattedDate }}</p>
        <div v-if="formattedMissingRange" class="missing-ranges">
          <div class="tooltip-container">
            <span class="missing-badge">
              {{ formattedMissingRange }}
            </span>
            <div class="tooltip">
              В периоде есть месяца без данных
              <!-- Месяцы без доступных данных: {{ formattedMissingRange }} -->
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

// Интерфейс Filters
interface Filters {
  from?: string | null;
  to?: string | null;
  metric?: 'count' | 'avg_duration';
  customer?: string | null; // <-- НОВОЕ ПОЛЕ
}

const props = defineProps<{
  filters: Filters;
  selectedPath?: string;
  missingMonths: string[];
}>();

// Форматирование дат в русский читаемый вид
const formattedDate = computed(() => {
  console.log('Current filters in formattedDate:', props.filters);
  const from = props.filters.from || '2025-01-01';
  const to = props.filters.to || '2025-12-31';

  const fromDate = new Date(from);
  const toDate = new Date(to);

  const fromFormatted = fromDate.toLocaleDateString('ru-RU', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
  const toFormatted = toDate.toLocaleDateString('ru-RU', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

  return `${fromFormatted} - ${toFormatted}`;
});

// Форматирование пропущенных месяцев как "первый - последний"
const formattedMissingRange = computed(() => {
  if (props.missingMonths.length === 0) return '';
  const sorted = [...props.missingMonths].sort();
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  return first === last ? first : `${first} - ${last}`;
});
</script>

<style scoped>
.selected-params {
  margin: 16px 0;
  width: 100%;

}

.params-content {
  backdrop-filter: blur(5px);
  border: 1px solid rgba(255, 193, 7, 0.3);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-sizing: border-box;
  min-height: 110px;
}

.customer-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  flex: 1;
}

.customer-name-wrapper {
  /* Стили скопированы из .logo-container для эффекта Glassmorphism */

  /* Фон, рамка и размытие */
  background: linear-gradient(to right, rgba(60, 60, 60, 0.2), rgba(30, 30, 30, 0.2));
  border-radius: 16px; /* Скругление */
  padding: 12px 24px; /* Увеличенные паддинги */
  border: 1px solid rgba(120, 120, 120, 0.2); /* Тонкая серая рамка */

  /* Эффект размытия фона (backdrop-filter) */
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);

  margin-top: 4px;
  transition: all 0.3s ease;

  /* Дополнительные свойства для центрирования содержимого (если нужно) */
  display: inline-flex;
  justify-content: center;
  align-items: center;
}

.customer-name {
  /* Стиль текста: жирный, белый */
  font-size: 1.5rem;
  line-height: 1.75rem;
  font-weight: 700; /* Очень жирный, как в логотипе */
  margin: 0;
  white-space: nowrap;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;

  /* ⚠️ ВОЗВРАТ К БЕЛОМУ ЦВЕТУ: УДАЛЕНЫ ВСЕ СВОЙСТВА ГРАДИЕНТА */
  color: #ffffff;
  background: none;
  -webkit-background-clip: unset;
  -webkit-text-fill-color: unset;
  margin: 0 10px;
}

.region-group {
  display: flex;
  align-items: center;
}

.icon-wrapper {
  width: 48px;
  height: 48px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #ffc107, #ff9800);
  color: #000000;
  margin-right: 12px;
}

.geo-icon {
  width: 24px;
  height: 24px;
}

.region-info {
  display: flex;
  flex-direction: column;
}

.label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #aaaaaa;
  margin: 0;
  margin-bottom: 2px;
}

.region-name {
  font-size: 1.5rem;
  line-height: 1.75rem;
  font-weight: 700;
  background: linear-gradient(to right, #ffd54f, #ff8f00);
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
  margin: 0;
}

.date-group {
  text-align: right;
}

.date-value {
  font-size: 1.125rem;
  font-weight: 500;
  color: #ffffff;
  margin: 0;
  margin-top: 2px;
}

.missing-ranges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
  justify-content: flex-end;
}

.tooltip-container {
  position: relative;
  display: inline-block;
}

.missing-badge {
  background-color: #a62828;
  color: #ffffff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.875rem;
  white-space: nowrap;
  transition: background-color 0.2s ease;
  cursor: pointer;
}

.missing-badge:hover {
  background-color: #b53b3b;
}

.tooltip {
  position: absolute;
  bottom: calc(0% - 40px);
  left: 50%;
  transform: translateX(-50%);
  background-color: rgba(0, 0, 0, 0.8);
  color: #ffffff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.875rem;
  white-space: nowrap;
  display: none;
  transition: none;
  pointer-events: none;
  z-index: 10;
}

.tooltip-container:hover .tooltip {
  display: block;
}
</style>
