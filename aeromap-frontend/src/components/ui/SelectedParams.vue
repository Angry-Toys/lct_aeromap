<template>
  <div class="selected-params">
    <div class="params-content">
      <div class="region-group">
        <div class="icon-wrapper">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            class="geo-icon"
          >
            <path
              fill-rule="evenodd"
              d="M11.54 22.351l.01.01a.75.75 0 001.9 0l.01-.01c.97-.97 2.65-3.19 3.92-6.08a17.39 17.39 0 001.74-6.86c0-4.69-3.81-8.5-8.5-8.5s-8.5 3.81-8.5 8.5c0 2.57.93 4.93 2.47 6.74 1.27 2.89 2.95 5.11 3.92 6.08zm.96-18.1a4 4 0 100 8 4 4 0 000-8z"
              clip-rule="evenodd"
            />
          </svg>
        </div>
        <div class="region-info">
          <p class="label">Выбранный регион</p>
          <h2 class="region-name">{{ selectedRegion || 'Russian Federation' }}</h2>
        </div>
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
}

const props = defineProps<{
  filters: Filters;
  selectedRegion?: string;
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
