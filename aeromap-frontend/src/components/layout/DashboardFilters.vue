<template>
  <div class="filters-panel">
    <h3 class="panel-title">Параметры Аналитики</h3>

    <div class="filter-section">
      <span class="section-label">Общая статистика</span>
      <div class="filter-group">
        <label for="year-select">Год:</label>
        <select id="year-select" v-model="filters.year">
          <option>2025</option>
          <option>2024</option>
          <option>2023</option>
        </select>
      </div>
      <div class="filter-group">
        <label for="month-select">Месяц:</label>
        <select id="month-select" v-model="filters.month">
          <option :value="null">Все</option>
          <option v-for="month in months" :key="month.value" :value="month.value">
            {{ month.name }}
          </option>
        </select>
      </div>
    </div>

    <div class="filter-section">
      <span class="section-label">Рейтинг регионов</span>
      <div class="filter-group">
        <label for="metric-select">Метрика:</label>
        <select id="metric-select" v-model="filters.metric">
          <option value="count">Количество полетов</option>
          <option value="avg_duration">Средняя длительность</option>
        </select>
      </div>
    </div>

    <button class="apply-btn" @click="applyFilters">Применить</button>
  </div>
</template>

<script setup lang="ts">
import { reactive } from 'vue';

const emit = defineEmits(['filters-updated']);

// Единый реактивный объект для всех фильтров
const filters = reactive({
  year: '2025',
  month: null as string | null, // Месяц может быть не выбран
  // from и to теперь не нужны, так как они дублируют год/месяц
  metric: 'count',
});

const months = [
  { value: '01', name: 'Январь' }, { value: '02', name: 'Февраль' },
  { value: '03', name: 'Март' }, { value: '04', name: 'Апрель' },
  { value: '05', name: 'Май' }, { value: '06', name: 'Июнь' },
  { value: '07', name: 'Июль' }, { value: '08', name: 'Август' },
  { value: '09', name: 'Сентябрь' }, { value: '10', name: 'Октябрь' },
  { value: '11', name: 'Ноябрь' }, { value: '12', name: 'Декабрь' },
];

const applyFilters = () => {
  // Отправляем все фильтры родительскому компоненту
  emit('filters-updated', filters);
};
</script>

<style scoped>
.filters-panel {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 16px 24px;
  background-color: #0f2346;
  border: 1px solid #226bcb;
  border-radius: 12px;
  margin-bottom: 24px;
}
.panel-title {
  color: #fff;
  font-size: 1.1rem;
  font-weight: 600;
  margin-right: 20px;
}
.filter-section {
  display: flex;
  align-items: center;
  gap: 16px;
  border-left: 1px solid #226bcb;
  padding-left: 20px;
}
.section-label {
  color: #a0c3ff;
  font-size: 0.8rem;
  text-transform: uppercase;
}
.filter-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
label {
  color: #a0c3ff;
  font-size: 0.9rem;
}
select {
  background: #0a1929;
  border: 1px solid #226bcb;
  border-radius: 8px;
  color: #fff;
  padding: 8px 12px;
  font-size: 0.9rem;
}
.apply-btn {
  margin-left: auto;
  padding: 10px 22px;
  font-size: 0.9rem;
  font-weight: bold;
  color: #fff;
  background-color: #30ceda;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.apply-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(48, 206, 218, 0.4);
}
</style>
