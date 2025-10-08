<template>
  <div class="chart-container">
    <div v-if="isLoading" class="status-loading">Загрузка данных по регионам...</div>
    <div v-else-if="error" class="status-loading error">{{ error }}</div>
    <v-chart v-else :option="chartOption" autoresize />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import VChart from 'vue-echarts';
import axios from 'axios';
import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, TitleComponent, VisualMapComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([BarChart, GridComponent, TooltipComponent, TitleComponent, VisualMapComponent, CanvasRenderer]);

// Типизация для пропсов
interface Filters {
  year: string;
  month: string | null;
  metric: 'count' | 'avg_duration';
}

const props = defineProps<{
  filters: Filters;
}>();

const isLoading = ref(true);
const error = ref('');
const chartOption = ref({});

const fetchData = async () => {
  isLoading.value = true;
  error.value = '';
  try {
    // Форматирование дат для API /api/regions/flights
    // Используем фильтр по году (игнорируем месяц, т.к. этот эндпоинт не поддерживает месяц)
    const fromDate = `${props.filters.year}-01-01`;
    const toDate = `${props.filters.year}-12-31`;

    const response = await axios.get('http://localhost:5000/api/regions/flights', {
      params: {
        from: fromDate,
        to: toDate,
        metric: props.filters.metric,
      }
    });

    const data = response.data.slice(0, 10); // Берем топ-10

    if (!data || data.length === 0) {
      error.value = 'Нет данных для отображения рейтинга регионов';
      chartOption.value = {};
      return;
    }

    const titleText = props.filters.metric === 'count'
      ? 'Топ 10 регионов по количеству полетов'
      : 'Топ 10 регионов по средней длительности полетов (мин)';

    chartOption.value = {
      title: {
        text: titleText,
        textStyle: {
          color: '#e0e0e0',
          fontWeight: 'normal',
          fontSize: 16
        },
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        textStyle: { color: '#fff' }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        axisLine: {
          lineStyle: { color: '#aaaaaa' } // Серые оси
        },
        splitLine: {
          lineStyle: { color: '#333333' } // Темно-серые фоновые линии
        },
        axisLabel: { color: '#aaaaaa' }
      },
      yAxis: {
        type: 'category',
        data: data.map((item: any) => item.name).reverse(),
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#e0e0e0' } // Белые метки категорий
      },
      series: [{
        type: 'bar',
        name: props.filters.metric === 'count' ? 'Полетов' : 'Длительность, мин',
        data: data.map((item: any) => item.value).reverse(),
        itemStyle: {
          // ЯНТАРНЫЙ ГРАДИЕНТ
          color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [
            { offset: 0, color: '#ffac30' },
            { offset: 1, color: '#ffc107' }
          ])
        },
        barMaxWidth: 20
      }]
    };
  } catch (e) {
    error.value = 'Ошибка при загрузке данных рейтинга';
    console.error(e);
  } finally {
    isLoading.value = false;
  }
};

// Наблюдаем за изменением фильтров и перезагружаем данные
watch(() => props.filters, fetchData, { deep: true, immediate: true });
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
.status-loading.error {
    color: #ff6666;
}
</style>
