<template>
  <div class="chart-container">
    <div v-if="isLoading" class="status-loading">Загрузка данных об активности...</div>
    <div v-else-if="!props.data" class="status-loading">Нет данных для отображения</div>
    <v-chart v-else-if="chartOption" :option="chartOption" autoresize />
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';

// --- ОСНОВНЫЕ ИМПОРТЫ ---\
import VChart from 'vue-echarts';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

// --- РЕГИСТРАЦИЯ КОМПОНЕНТОВ ---\
echarts.use([
  LineChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  CanvasRenderer
]);

interface HourlyDataPoint {
    hour: number;
    count: number;
}

const props = defineProps<{
    data: HourlyDataPoint[] | null;
    isLoading: boolean;
}>();

const chartOption = ref({});

const updateChart = () => {
  if (!props.data || props.data.length === 0) {
    chartOption.value = {};
    return;
  }

  // Сортируем данные по часу и заполняем массивы
  const sortedData = [...props.data].sort((a, b) => a.hour - b.hour);
  const hours = sortedData.map(item => item.hour);
  const counts = sortedData.map(item => item.count);

  chartOption.value = {
    title: {
      text: 'Среднесуточная динамика полетов (по часам)',
      textStyle: { color: '#e0e0e0', fontWeight: 'normal', fontSize: 16 },
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      textStyle: { color: '#fff' },
      formatter: (params: any) => {
        const item = params[0];
        return `Час ${item.name}:00 <br/> Полетов: ${item.value}`;
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: hours,
      axisLine: { lineStyle: { color: '#aaaaaa' } },
      axisLabel: { color: '#aaaaaa' }
    },
    yAxis: {
      type: 'value',
      splitLine: {
        lineStyle: {
          color: '#333333' // Темно-серые фоновые линии
        }
      },
      axisLabel: { color: '#aaaaaa' }
    },
    series: [{
      name: 'Кол-во полетов',
      type: 'line',
      smooth: true,
      data: counts,
      // ЯНТАРНЫЙ ЦВЕТ ЛИНИИ
      itemStyle: { color: '#ffc107' },
      lineStyle: { width: 3 },
      // ЯНТАРНАЯ ЗАЛИВКА
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(255, 193, 7, 0.5)' },
          { offset: 1, color: 'rgba(255, 193, 7, 0)' }
        ])
      }
    }]
  };
};

watch(() => props.data, updateChart, { immediate: true });
</script>

<style scoped>
.chart-container {
  /* Фон контейнера: Чистый чёрный */
  background-color: #000000;
  border: 1px solid #333333;
  border-radius: 12px;
  padding: 16px;
  height: 400px;
  position: relative;
}
.status-loading {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #aaaaaa;
  font-size: 1rem;
  background-color: rgba(0, 0, 0, 0.5);
  border-radius: 12px;
  z-index: 5;
}
</style>
