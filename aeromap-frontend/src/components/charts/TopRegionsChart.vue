<template>
  <div class="chart-container">
    <v-chart v-if="!isLoading" :option="chartOption" autoresize />
    <div v-if="isLoading" class="status-loading">Загрузка...</div>
    <div v-if="error" class="status-loading">{{ error }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import VChart from 'vue-echarts';
import axios from 'axios';
import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, TitleComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([BarChart, GridComponent, TooltipComponent, TitleComponent, CanvasRenderer]);

const props = defineProps<{
  filters: { year: string; month: string | null };
}>();

const isLoading = ref(true);
const error = ref('');
const chartOption = ref({});

const fetchData = async () => {
  isLoading.value = true;
  error.value = '';
  try {
    // Используем правильный эндпоинт /api/regions/flights
    const response = await axios.get('http://localhost:5000/api/regions/flights', {
      params: {
        // Преобразуем фильтры в нужный формат from/to
        from: `${props.filters.year}-01-01`,
        to: `${props.filters.year}-12-31`,
        metric: 'count'
      }
    });
    const data = response.data;

    if (!data || data.length === 0) {
      error.value = 'Нет данных для отображения';
      return;
    }

    chartOption.value = {
      title: { text: 'Топ регионов по активности', textStyle: { color: '#fff', fontWeight: 'normal' } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'value', axisLine: { lineStyle: { color: '#888' } }, splitLine: { lineStyle: { color: '#2a3b5a' }} },
      yAxis: { type: 'category', data: data.map((item: any) => item.name).reverse(), axisLine: { show: false }, axisTick: { show: false } },
      series: [{
        type: 'bar',
        data: data.map((item: any) => item.value).reverse(),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [
            { offset: 0, color: '#0e4a91' },
            { offset: 1, color: '#30ceda' }
          ])
        },
        barMaxWidth: 30
      }]
    };
  } catch (e) {
    error.value = 'Ошибка при загрузке данных';
    console.error(e);
  } finally {
    isLoading.value = false;
  }
};

onMounted(fetchData);
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 100%;
  padding: 20px;
  background: #0f2346;
  border: 1px solid #226bcb;
  border-radius: 12px;
  position: relative;
  min-height: 300px; /* Минимальная высота для предотвращения коллапса */
}
.status-loading {
  position: absolute;
  inset: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #a0c3ff;
}
</style>
