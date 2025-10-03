<template>
  <div class="chart-container">
    <v-chart v-if="!isLoading && chartOption" :option="chartOption" autoresize />
    <div v-if="isLoading" class="status-loading">Загрузка данных об активности...</div>
    <div v-if="!isLoading && !props.data" class="status-loading">Нет данных для отображения</div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';

// --- ОСНОВНЫЕ ИМПОРТЫ ---
// Компонент для встраивания ECharts в Vue
import VChart from 'vue-echarts';
// Ядро ECharts, которое всегда необходимо
import * as echarts from 'echarts/core';

// --- ИМПОРТЫ КОМПОНЕНТОВ ECHARTS ---
// Импортируем только те графики, которые будем использовать (в данном случае, линейный и столбчатый)
import { LineChart, BarChart } from 'echarts/charts';
// Импортируем компоненты, необходимые для графика: заголовок, подсказки, сетка, легенда
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent
} from 'echarts/components';
// Импортируем "отрисовщик" (renderer), который будет рисовать график на холсте (canvas)
import { CanvasRenderer } from 'echarts/renderers';

// --- РЕГИСТРАЦИЯ КОМПОНЕНТОВ ---
// "Сообщаем" ядру ECharts, какие части мы будем использовать в этом компоненте
echarts.use([
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  LineChart,
  BarChart,
  CanvasRenderer
]);

// --- ЛОГИКА КОМПОНЕНТА ---

// Определяем входные параметры (props), которые компонент ожидает от родителя (DashboardView)
const props = defineProps<{
  data: { hour: number; count: number }[] | null;
  isLoading: boolean;
}>();

// Реактивная переменная для хранения опций графика
const chartOption = ref({});

// Функция для обновления конфигурации графика
const updateChart = () => {
  // Если данных нет, ничего не делаем
  if (!props.data || props.data.length === 0) {
    chartOption.value = {}; // Очищаем опции, чтобы график не показывал старые данные
    return;
  }

  // Готовим данные для осей графика
  const hours = props.data.map(item => `${item.hour}:00`);
  const counts = props.data.map(item => item.count);

  // Формируем полный объект опций для ECharts
  chartOption.value = {
    title: {
      text: 'Активность по часам',
      textStyle: { color: '#fff', fontWeight: 'normal' }
    },
    tooltip: {
      trigger: 'axis'
    },
    grid: { // Отступы для графика, чтобы подписи не обрезались
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: hours,
      axisLine: { lineStyle: { color: '#888' } }
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#2a3b5a' } } // Цвет фоновых линий сетки
    },
    series: [{
      name: 'Кол-во полетов',
      type: 'line', // Используем линейный график для наглядности динамики
      smooth: true, // Сглаживаем линию
      data: counts,
      itemStyle: { color: '#30ceda' },
      areaStyle: { // Заливка под графиком
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(48, 206, 218, 0.5)' },
          { offset: 1, color: 'rgba(48, 206, 218, 0)' }
        ])
      }
    }]
  };
};

// "Наблюдатель", который автоматически вызывает updateChart,
// как только данные (props.data) приходят или обновляются от родительского компонента
watch(() => props.data, updateChart, { immediate: true });
</script>

<style scoped>
/* Стили для контейнера графика, как и в других компонентах */
.chart-container {
  width: 100%;
  height: 100%;
  min-height: 300px;
  padding: 20px;
  background-color: #0f2346;
  border: 1px solid #226bcb;
  border-radius: 12px;
  position: relative;
}
.status-loading {
  position: absolute;
  inset: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #a0c3ff;
  font-size: 0.9rem;
}
</style>
