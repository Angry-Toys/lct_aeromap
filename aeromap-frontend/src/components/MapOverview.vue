<template>
  <div class="chart-wrapper">
    <div v-if="isLoading">Загрузка данных...</div>
    <v-chart
      v-else-if="option"
      ref="chartRef"
      :option="option"
      :init-opts="{ renderer: 'canvas' }"
      autoresize
      class="map-container"
    />
    <div v-if="errorMessage" style="color: red">{{ errorMessage }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue';
import * as echarts from 'echarts';
import VChart from 'vue-echarts';
import axios from 'axios';

const chartRef = ref(null);
const option = ref(null);
const errorMessage = ref('');
const isLoading = ref(false);
const flightData = ref<{ name: string; value: number }[]>([]);

onMounted(async () => {
  await nextTick();

  if (!chartRef.value) {
    errorMessage.value = 'Контейнер не найден.';
    return;
  }

  isLoading.value = true;
  try {
    // Запрос к API через Axios с прокси (/api -> localhost:5000)
    const response = await axios.get('/api/regions/flights', {
      params: {
        from: '2025-01-01',
        to: '2025-09-30', // Актуальная дата: 30.09.2025
        metric: 'count'
      }
    });

    flightData.value = response.data; // Предполагаем, что ответ - [{ name, value }]

    // Загрузка GeoJSON
    const geoResponse = await fetch('/maps/Russia.geojson');
    if (!geoResponse.ok) throw new Error(`HTTP: ${geoResponse.status}`);
    const russiaGeoJSON = await geoResponse.json();

    if (!russiaGeoJSON || russiaGeoJSON.type !== 'FeatureCollection') {
      throw new Error('Некорректный GeoJSON.');
    }

    echarts.registerMap('Russia', russiaGeoJSON);

    await new Promise(resolve => setTimeout(resolve, 100));

    option.value = {
      title: {
        text: 'Распределение полетов БПЛА по регионам РФ',
        left: 'center',
        textStyle: { color: '#fff', fontSize: 16 }
      },
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} полетов'
      },
      visualMap: {
        min: 0,
        max: Math.max(...flightData.value.map(d => d.value || 0)),
        text: ['Высокая активность', 'Низкая'],
        realtime: false,
        calculable: true,
        inRange: { color: ['#2d8aff', '#0062E8'] }
      },
      toolbox: {
        show: true,
        feature: {
          saveAsImage: { title: 'Экспорт в PNG', type: 'png' }
        }
      },
      series: [
        {
          name: 'Полеты',
          type: 'map',
          map: 'Russia',
          roam: true,
          zoom: 1,
          animation: false,
          silent: false,
          center: [90, 60],
          aspectScale: 0.75,
          boundingCoords: [[20, 40], [200, 80]],
          label: {
            normal: { show: false },
            emphasis: { show: true, color: '#fff', textStyle: { fontSize: 12 } }
          },
          itemStyle: {
            normal: {
              borderColor: '#2d8aff',
              areaColor: '#2d8aff',
              shadowColor: '#226bcb',
              shadowBlur: 10,
              shadowOffsetY: 10
            },
            emphasis: { areaColor: '#0062E8' }
          },
          data: flightData.value
        }
      ]
    };
  } catch (error) {
    errorMessage.value = `Ошибка при загрузке данных: ${error instanceof Error ? error.message : 'Неизвестная ошибка'}`;
    console.error('API Error:', error);
  } finally {
    isLoading.value = false;
  }
});
</script>

<style scoped>
.chart-wrapper {
  width: 80vw;
  height: 80vh;
  position: relative;
}
:deep(.map-container) {
  width: 100%;
  height: 100%;
  background: linear-gradient(to bottom, #2d8aff, #0062E8);
  border: 2px solid #30ceda;
  box-shadow: 0 0 15px rgba(34, 107, 203, 0.5);
  border-radius: 10px;
  overflow: hidden;
}
:deep(.echarts) {
  width: 100% !important;
  height: 100% !important;
}
</style>
