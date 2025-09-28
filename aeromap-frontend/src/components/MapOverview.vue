<template>
  <div class="chart-wrapper">
    <v-chart
      ref="chartRef"
      :option="option"
      :init-opts="{ renderer: 'canvas' }"
      autoresize
      class="map-container"
    />
    <div v-if="errorMessage" style="color: red;">{{ errorMessage }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue';
import * as echarts from 'echarts';  // Полный импорт по гайдам (без модульного use)
import VChart from 'vue-echarts';

const chartRef = ref(null);
const props = defineProps<{
  flightData: Array<{ name: string; value: number }>;
}>();

const option = ref(null);  // Без типов, если TS отключён
const errorMessage = ref('');

onMounted(async () => {
  console.log('onMounted triggered');
  await nextTick();

  console.log('chartRef after nextTick:', chartRef.value);
  if (!chartRef.value) {
    errorMessage.value = 'Контейнер не найден.';
    return;
  }

  let russiaGeoJSON: any;
  try {
    const response = await fetch('/maps/Russia.geojson');
    if (!response.ok) throw new Error(`HTTP: ${response.status}`);
    russiaGeoJSON = await response.json();

    if (!russiaGeoJSON || russiaGeoJSON.type !== 'FeatureCollection') {
      throw new Error('Некорректный GeoJSON.');
    }

    echarts.registerMap('Russia', russiaGeoJSON);  // Без типов GeoJSONSource

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
        max: Math.max(...props.flightData.map(d => d.value || 0)),
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
          roam: true,  // Зум и перемещение
          zoom: 1,
          animation: false,
          silent: false,  // Интерактив
          center: [90, 60],  // Центр на Россию
          aspectScale: 0.75, // Форма карты
          boundingCoords: [[20, 40], [200, 80]],  // Границы для Чукотки
          label: {
            normal: {
              show: false  // Без подписей по умолчанию
            },
            emphasis: {
              show: true,
              color: '#fff',
              textStyle: { fontSize: 12 }
            }
          },
          itemStyle: {
            normal: {
              borderColor: '#2d8aff',
              areaColor: '#2d8aff',
              shadowColor: '#226bcb',
              shadowBlur: 10,
              shadowOffsetY: 10
            },
            emphasis: {
              areaColor: '#0062E8'
            }
          },
          data: props.flightData
        }
      ]
    };
  } catch (error) {
    errorMessage.value = `Ошибка: ${error instanceof Error ? error.message : 'Неизвестная'}`;
    console.error(error);
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
