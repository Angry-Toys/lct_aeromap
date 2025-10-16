<template>
  <div class="modal-overlay">
    <div class="modal-content">
      <div class="modal-header">
        <h3>Динамика роста полётов по месяцам</h3>
        <button class="close-button" @click="$emit('close')">
          <svg viewBox="0 0 24 24" class="close-icon">
            <line x1="4" y1="4" x2="20" y2="20" />
            <line x1="4" y1="20" x2="20" y2="4" />
          </svg>
        </button>
      </div>
      <div v-if="props.data.length === 0" class="no-data">
        Нет данных для отображения
      </div>
      <v-chart v-else class="chart-container" :option="chartOption" autoresize />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import VChart from 'vue-echarts';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, GridComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

// Регистрация компонентов ECharts
echarts.use([LineChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent, CanvasRenderer]);

interface GrowthData {
  month: string;
  value: number;
}

const props = defineProps<{
  data: GrowthData[];
}>();

defineEmits(['close']);

const chartOption = computed(() => ({
  title: {
    text: '',
    textStyle: { color: '#e0e0e0', fontWeight: 'normal', fontSize: 16 },
    left: 'center',
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    textStyle: { color: '#fff' },
    formatter: (params: any) => {
      const item = params[0];
      const value = item.value >= 0 ? `+${item.value}` : `${item.value}`;
      return `Месяц: ${item.name}<br/>Рост: ${value}%`;
    },
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    data: props.data.map(item => item.month),
    axisLine: { lineStyle: { color: '#aaaaaa' } },
    axisLabel: { color: '#aaaaaa' },
  },
  yAxis: {
    type: 'value',
    splitLine: { lineStyle: { color: '#333333' } },
    axisLabel: { color: '#aaaaaa', formatter: '{value}%' },
  },
  series: [
    {
      name: 'Рост полётов',
      type: 'line',
      smooth: true,
      data: props.data.map(item => parseFloat(item.value.toFixed(1))),
      itemStyle: { color: '#047857' },
      lineStyle: { width: 3 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(4, 120, 87, 0.5)' }, // #047857
          { offset: 1, color: 'rgba(6, 95, 70, 0)' }, // #065f46
        ]),
      },
    },
  ],
}));
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(5px);
  z-index: 200;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px;
}

.modal-content {
  background: #000000;
  border: 1px solid #333333;
  border-radius: 16px;
  padding: 32px;
  max-width: 896px;
  width: 100%;
  max-height: 80vh;
  overflow-y: auto;
  animation: fadeInScale 0.1s ease-out forwards;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0; /* Убран отступ */
  text-align: center;
}

.modal-header h3 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #ffffff;
  flex: 1;
  text-align: center;
}

.close-button {
  background: none;
  border: none;
  cursor: pointer;
  transition: filter 0.3s ease;
}

.close-button:hover {
  filter: brightness(1.2);
}

.close-icon {
  width: 20px;
  height: 20px;
  stroke: #aaaaaa;
  stroke-width: 2;
}

.close-button:hover .close-icon {
  stroke: #ffffff;
}

.no-data {
  color: #aaaaaa;
  text-align: center;
  font-size: 1rem;
}

.chart-container {
  width: 100%;
  height: 384px;
}

@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}
</style>
