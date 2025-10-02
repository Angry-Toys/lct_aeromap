<template>
  <div class="chart-wrapper">

    <div v-if="isLoading" class="status-overlay">
      <p>Загрузка карты...</p>
    </div>

    <div v-else-if="errorMessage" class="status-overlay error">
      <p>Не удалось загрузить данные карты.</p>
      <p class="error-details">{{ errorMessage }}</p>
      <button @click="fetchData">Попробовать снова</button>
    </div>

    <v-chart
      v-else
      ref="chartRef"
      :option="option"
      :init-opts="{ renderer: 'canvas' }"
      autoresize
      class="map-container"
    />

  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue';
import * as echarts from 'echarts';
import VChart from 'vue-echarts';
import axios from 'axios';

// --- Состояние компонента ---
const chartRef = ref(null);
const option = ref(null);
const errorMessage = ref('');
const isLoading = ref(false);
const flightData = ref([]);

// --- Логика загрузки данных ---
const fetchData = async () => {
  isLoading.value = true;
  errorMessage.value = ''; // Сбрасываем предыдущую ошибку
  option.value = null; // Очищаем старые данные карты

  try {
    console.log('Начинаем загрузку данных...');

    // Загрузка данных о полетах
    const response = await axios.get('http://localhost:5000/api/regions/flights', {
      params: {
        from: '2025-01-01',
        to: '2025-09-30',
        metric: 'count'
      }
    });
    flightData.value = response.data;
    console.log('Данные из API получены:', flightData.value);

    // Загрузка GeoJSON файла карты
    const geoResponse = await fetch('/maps/Russia.geojson');
    if (!geoResponse.ok) {
      throw new Error(`Ошибка загрузки GeoJSON: HTTP ${geoResponse.status}`);
    }
    const russiaGeoJSON = await geoResponse.json();
    if (!russiaGeoJSON || russiaGeoJSON.type !== 'FeatureCollection') {
      throw new Error('Некорректный формат GeoJSON.');
    }
    console.log('GeoJSON загружен успешно.');

    // Регистрация карты в ECharts
    echarts.registerMap('Russia', russiaGeoJSON);

    // --- Формирование опций для карты ---
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
        inRange: { color: ['#2d8aff', '#0062E8'] },
        textStyle: { color: '#fff' }
      },
      toolbox: {
        show: true,
        orient: 'vertical',
        left: 'right',
        top: 'center',
        feature: {
          saveAsImage: { title: 'Сохранить как PNG', backgroundColor: '#0e4a91' }
        },
        iconStyle: {
          borderColor: '#fff'
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
            show: false,
            emphasis: { show: true, color: '#fff' }
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

    console.log('Карта успешно инициализирована.');
  } catch (error) {
    errorMessage.value = error.message || 'Произошла неизвестная ошибка.';
    console.error('Подробности ошибки:', error);
    // Дополнительная информация для отладки
    if (error.response) {
      console.error('API статус:', error.response.status, 'Данные:', error.response.data);
    } else if (error.request) {
      console.error('Нет ответа от сервера. Проверьте API и сетевое соединение.');
    }
  } finally {
    isLoading.value = false;
  }
};

// --- Хук жизненного цикла ---
onMounted(() => {
  fetchData();
});
</script>

<style scoped>
.chart-wrapper {
  width: 80vw;
  height: 80vh;
  position: relative;
  background: linear-gradient(to bottom, #4abef8, #0062E8);
  border: 2px solid #30ceda;
  box-shadow: 0 0 15px rgba(34, 107, 203, 0.5);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
}

.map-container,
:deep(.echarts) {
  width: 100% !important;
  height: 100% !important;
}

/* Стили для оверлеев состояния */
.status-overlay {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 20px;
  text-align: center;
  color: #fff;
  font-size: 1.2rem;
}

.status-overlay.error {
  color: #ffdddd;
}

.error-details {
  font-size: 0.9rem;
  color: #ffb8b8;
  margin-top: 5px;
  max-width: 80%;
}

.status-overlay button {
  margin-top: 20px;
  padding: 10px 20px;
  font-size: 1rem;
  color: #fff;
  background-color: #30ceda;
  border: 1px solid #fff;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.status-overlay button:hover {
  background-color: #26a2aa;
}
</style>
