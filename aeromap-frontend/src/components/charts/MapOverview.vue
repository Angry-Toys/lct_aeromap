<template>
  <div class="chart-wrapper">

    <div v-if="isLoading" class="status-overlay">
      <div class="spinner"></div>
      <p>Загрузка данных...</p>
    </div>

    <div v-if="errorMessage" class="status-overlay error">
      <p>😕 Ошибка при загрузке данных</p>
      <p class="error-details">{{ errorMessage }}</p>
      <button @click="fetchFlightData">Попробовать снова</button>
    </div>

    <v-chart
      v-if="isMapReady"
      ref="chartRef"
      :option="option"
      :autoresize="true"
      class="map-container"
      @click="handleChartClick"
    />

    <button v-if="isMapReady" class="refresh-button" @click="refreshMap" data-html2canvas-ignore="true">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="refresh-icon">
        <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 010-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd" />
      </svg>
      <span>Центрировать</span>
    </button>

    <div v-else class="status-overlay">
      <p>Инициализация карты...</p>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import * as echarts from 'echarts/core';
import { MapChart, EffectScatterChart } from 'echarts/charts';
import { TooltipComponent, VisualMapComponent, GeoComponent, TitleComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import VChart from 'vue-echarts';
import axios from 'axios';
import api from '@/utils/api';  // Импорт глобального api с токеном

// Регистрация необходимых компонентов ECharts
echarts.use([
  MapChart,
  EffectScatterChart,
  TooltipComponent,
  VisualMapComponent,
  GeoComponent,
  TitleComponent,
  CanvasRenderer
]);

// --- Состояния компонента ---
const isLoading = ref(false);
const isMapReady = ref(false);
const errorMessage = ref('');
const chartRef = ref<InstanceType<typeof VChart> | null>(null);
//const chartRef = ref(null);
const option = ref({});

defineExpose({
  getEChartsInstance: () => chartRef.value?.$el ? echarts.getInstanceByDom(chartRef.value.$el) : null
});

const props = defineProps<{
  filters: { from?: string | null; to?: string | null; metric?: 'count' | 'avg_duration' };
}>();

const emit = defineEmits(['region-selected']);

// --- Базовая опция для карты (пустое состояние) ---
const getBaseMapOption = (mapName: string) => ({
  geo: {
    map: mapName,
    roam: true,
    zoom: 1.0,
    center: [100, 60],
    scaleLimit: { min: 0.8, max: 8 },
    boundingCoords: [[30, 40], [180, 82]],
    itemStyle: {
      areaColor: '#1a1a1a',
      borderColor: '#000000',
      borderWidth: 1,
    },
    emphasis: {
      itemStyle: {
        areaColor: '#333333',
        borderColor: '#ffc107',
        borderWidth: 2,
      },
      label: {
        show: false,
        color: '#e0e0e0',
        fontSize: 14,
        fontWeight: '500',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: [4, 8],
        borderRadius: 4
      }
    },
    select: {
      itemStyle: {
        areaColor: '#4f46e5', // Индиго для выделенного региона
        borderColor: '#ffd54f', // Светлый янтарный для границы
        borderWidth: 2,
      },
      label: {
        show: true,
        color: '#e0e0e0', // Мягкий белый для надписи
        fontSize: 14,
        fontWeight: '500',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: [4, 8],
        borderRadius: 4
      }
    }
  },
  tooltip: {
    trigger: 'item',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    textStyle: { color: '#ffc107' },
    formatter: '{b}: <strong>{c}</strong> полетов'
  },
  series: []
});

// --- Обработчик клика по карте ---
const handleChartClick = (params: any) => {
  if (params.name && params.seriesType === 'map') {
    emit('region-selected', params.name);
    console.log('Region selected:', params.name);
  }
};

// --- Функция №1: Инициализация карты ---
const initMap = async () => {
  try {
    const geoResponse = await axios.get('/maps/Russia.geojson');
    console.log('GeoJSON loaded:', geoResponse.data);
    echarts.registerMap('Russia', geoResponse.data);

    option.value = getBaseMapOption('Russia');
    isMapReady.value = true;

    await fetchFlightData();
  } catch (error) {
    console.error('Критическая ошибка: не удалось загрузить GeoJSON карты.', error);
    errorMessage.value = 'Не удалось загрузить файл карты. Проверьте путь и доступность файла.';
  }
};

// --- Функция №2: Загрузка данных о полётах ---
const fetchFlightData = async () => {
  isLoading.value = true;
  try {
    const response = await api.get('/api/regions/flights', {
      params: {
        from: props.filters.from || '2025-01-01',
        to: props.filters.to || '2025-12-31',
        metric: props.filters.metric || 'count',
      }
    });

    const flightData = response.data;
    console.log('Flight data received:', flightData);

    if (chartRef.value) {
      if (flightData && flightData.length > 0) {
        const maxValue = Math.max(...flightData.map((item: { value: number }) => item.value));

        chartRef.value.setOption({
          tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            textStyle: { color: '#ffc107' },
            formatter: '{b}: <strong>{c}</strong> полетов'
          },
          visualMap: {
            min: 0,
            max: maxValue,
            left: 'left',
            top: 'top', // Перенос в левый верхний угол
            orient: 'horizontal', // Горизонтальное положение
            padding: [20, 0, 0, 20], // Отступы: сверху 10px, слева 10px
            text: ['Макс.', 'Мин.'],
            inRange: {
              color: ['#1a1a1a', '#ffc107']
            },
            calculable: true,
            textStyle: { color: '#e0e0e0' }
          },
          series: [{
            name: 'Активность полетов',
            type: 'map',
            geoIndex: 0,
            data: flightData
          }]
        });
      } else {
        chartRef.value.setOption(getBaseMapOption('Russia'));
      }
    }
  } catch (error: any) {
    errorMessage.value = error.response?.data?.error || 'Неизвестная ошибка API.';
    console.error('Ошибка при загрузке данных о полётах:', error);
  } finally {
    isLoading.value = false;
  }
};

// --- Функция обновления/центрирования карты ---
const refreshMap = () => {
  if (chartRef.value) {
    chartRef.value.setOption({
      geo: {
        zoom: 1.2,
        center: [95, 65]
      }
    });
    fetchFlightData(); // Перезагрузка данных
  }
};

// --- Хук жизненного цикла ---
onMounted(() => {

  initMap();
});
</script>

<style scoped>
.chart-wrapper {
  /* width: 100%; */
  height: 60vh;
  position: relative;
  background-color: #000000;
  border: 1px solid #333333;
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #fff;
  margin: 20px 0; /* Добавлен отступ сверху и снизу */

}


.map-container {
  width: 100%; /* Полная ширина */
  height: 100%;
}

/* Оверлей для статусов загрузки и ошибок */
.status-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.9);
  backdrop-filter: blur(5px);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 10;
  text-align: center;
}

.status-overlay p {
  font-size: 1.1rem;
}

.status-overlay.error {
  color: #ff6666;
}

.error-details {
  font-size: 0.9rem;
  color: #aaaaaa;
  margin-top: 8px;
}

.status-overlay button {
  margin-top: 20px;
  padding: 10px 20px;
  background-color: #ffc107;
  border: none;
  border-radius: 5px;
  color: #000000;
  cursor: pointer;
  transition: background-color 0.2s;
}

.status-overlay button:hover {
  background-color: #ffac30;
}

.refresh-button {
  position: absolute;
  top: 20px;
  right: 20px;
  margin-left: auto;
  padding: 8px 16px;
  font-size: 0.8rem;
  font-weight: bold;
  color: var(--vt-c-black, #000000);
  background: linear-gradient(to right, var(--color-heading, #ffc107), #ff9800);
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  transform: translateY(-2px);
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: 8px;
  z-index: 10;
}

.refresh-button:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(255, 193, 7, 0.4); /* Свечение при наведении */
}

.refresh-icon {
  width: 16px;
  height: 16px;
  color: var(--vt-c-black, #000000);
}

/* Анимация спиннера */
.spinner {
  border: 4px solid rgba(255, 255, 255, 0.2);
  border-left-color: #ffc107;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  margin-bottom: 15px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
