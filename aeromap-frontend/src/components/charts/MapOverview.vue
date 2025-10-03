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
    />

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
const isLoading = ref(false); // Для индикации загрузки именно ДАННЫХ, а не карты
const isMapReady = ref(false); // Флаг, что GeoJSON загружен и карта готова к отображению
const errorMessage = ref('');
const chartRef = ref<InstanceType<typeof VChart> | null>(null);
const option = ref({});

// --- Базовая опция для карты (пустое состояние) ---
const getBaseMapOption = (mapName: string) => ({
  title: {
    text: 'Карта Активности Полётов',
    subtext: 'Ожидание данных...',
    left: 'center',
    textStyle: { color: '#e0e0e0' },
    subtextStyle: { color: '#888' }
  },
  geo: { // Используем geo для более гибкого управления слоями
    map: mapName,
    roam: true,
    zoom: 1.2,
    center: [95, 65],
    itemStyle: {
      areaColor: '#1d3a5e', // Нейтральный цвет регионов без данных
      borderColor: '#4270a1',
      borderWidth: 1,
    },
    emphasis: {
      itemStyle: {
        areaColor: '#30ceda', // Цвет при наведении
      },
      label: {
        show: false
      }
    }
  },
  series: [] // Серии данных изначально пусты
});


// --- Функция №1: Инициализация карты ---
const initMap = async () => {
  try {
    const geoResponse = await axios.get('/maps/Russia.geojson');
    echarts.registerMap('Russia', geoResponse.data);

    option.value = getBaseMapOption('Russia');
    isMapReady.value = true;

    // Сразу после инициализации карты запрашиваем данные о полётах
    await fetchFlightData();

  } catch (error) {
    console.error("Критическая ошибка: не удалось загрузить GeoJSON карты.", error);
    errorMessage.value = "Не удалось загрузить файл карты. Проверьте путь и доступность файла.";
  }
};

// --- Функция №2: Загрузка и обновление данных о полётах ---
const fetchFlightData = async () => {
  isLoading.value = true;
  errorMessage.value = '';

  try {
    const response = await axios.get('http://localhost:5000/api/regions/flights');
    const flightData = response.data;

    // Обновляем ECharts только если он уже инициализирован
    if (chartRef.value) {
      if (flightData && flightData.length > 0) {
        // Данные есть - обновляем опции для отображения
        const maxValue = Math.max(...flightData.map((item: {value: number}) => item.value));

        chartRef.value.setOption({
          title: {
            subtext: 'Данные успешно загружены' // Меняем подзаголовок
          },
          tooltip: {
            trigger: 'item',
            formatter: '{b}: <strong>{c}</strong> полетов'
          },
          visualMap: {
            min: 0,
            max: maxValue,
            left: 'left',
            top: 'bottom',
            text: ['Макс.', 'Мин.'],
            inRange: {
              color: ['#30ceda', '#226bcb', '#0e4a91']
            },
            calculable: true,
            textStyle: { color: '#ccc' }
          },
          series: [{
            name: 'Активность полетов',
            type: 'map',
            geoIndex: 0, // Привязываем серию к geo-компоненту
            data: flightData
          }]
        });
      } else {
        // Данных нет - возвращаем карту в базовое состояние
        chartRef.value.setOption(getBaseMapOption('Russia'));
      }
    }
  } catch (error: any) {
    errorMessage.value = error.response?.data?.error || "Неизвестная ошибка API.";
    console.error("Ошибка при загрузке данных о полётах:", error);
  } finally {
    isLoading.value = false;
  }
};

// --- Хук жизненного цикла ---
onMounted(() => {
  initMap();
});
</script>

<style scoped>
.chart-wrapper {
  width: 80vw;
  height: 80vh;
  position: relative;
  background-color: #0a1929;
  border: 1px solid #226bcb;
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #fff;
}

.map-container {
  width: 100%;
  height: 100%;
}

/* Оверлей для статусов загрузки и ошибок */
.status-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(10, 25, 41, 0.7);
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
  color: #ff8e8e;
}

.error-details {
  font-size: 0.9rem;
  color: #a0a0a0;
  margin-top: 8px;
}

.status-overlay button {
  margin-top: 20px;
  padding: 10px 20px;
  background-color: #30ceda;
  border: none;
  border-radius: 5px;
  color: #fff;
  cursor: pointer;
  transition: background-color 0.2s;
}

.status-overlay button:hover {
  background-color: #2ab8c5;
}

/* Анимация спиннера */
.spinner {
  border: 4px solid rgba(255, 255, 255, 0.2);
  border-left-color: #30ceda;
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
