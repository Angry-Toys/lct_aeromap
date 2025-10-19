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

    <template v-if="isMapReady">
      <v-chart
        ref="chartRef"
        :option="option"
        :autoresize="true"
        class="map-container"
        @click="handleChartClick"
      />

      <button class="refresh-button" @click="refreshMap" data-html2canvas-ignore="true">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="refresh-icon">
          <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 010-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd" />
        </svg>
        <span>Центрировать</span>
      </button>

      <button v-if="currentView === 'region'" class="back-button" @click="goBack" data-html2canvas-ignore="true">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="back-icon">
          <path fill-rule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clip-rule="evenodd" />
        </svg>
        <span>Назад к регионам</span>
      </button>
    </template>

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
import { HeatmapChart } from 'echarts/charts';
import { ScatterChart } from 'echarts/charts';
import { CanvasRenderer } from 'echarts/renderers';
import VChart from 'vue-echarts';
import axios from 'axios';
import api from '@/utils/api';

echarts.use([
  MapChart,
  EffectScatterChart,
  TooltipComponent,
  VisualMapComponent,
  GeoComponent,
  TitleComponent,
  CanvasRenderer,
  HeatmapChart,
  ScatterChart
]);


function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}


const isLoading = ref(false);
const isMapReady = ref(false);
const errorMessage = ref('');
const chartRef = ref<InstanceType<typeof VChart> | null>(null);
const option = ref({});
const currentView = ref<'country' | 'region'>('country');
const selectedRegion = ref<string | null>(null);
const selectedDistrict = ref<string | null>(null);
const cachedGeo = ref<Record<string, any>>({});

const calculateCentroid = (geoJson: any): [number, number] => {
  let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
  geoJson.features.forEach((feature: any) => {
    const coordinates = feature.geometry.coordinates;
    const flattenCoords = (coords: any) => {
      if (typeof coords[0] === 'number') {
        minLon = Math.min(minLon, coords[0]);
        maxLon = Math.max(maxLon, coords[0]);
        minLat = Math.min(minLat, coords[1]);
        maxLat = Math.max(maxLat, coords[1]);
      } else {
        coords.forEach(flattenCoords);
      }
    };
    flattenCoords(coordinates);
  });
  const centerLon = (minLon + maxLon) / 2;
  const centerLat = (minLat + maxLat) / 2;
  return [centerLon, centerLat];
};

const pointInPolygon = (point: [number, number], ring: [number, number][]): boolean => {
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
};

const randomPointInPolygon = (polygon: any): [number, number] => {
  // Handle Polygon/MultiPolygon
  let rings = [];
  if (polygon.type === 'Polygon') {
    rings = polygon.coordinates;  // array of rings
  } else if (polygon.type === 'MultiPolygon') {
    rings = polygon.coordinates.flat();  // flat to rings
  } else {
    return [0, 0];  // Fallback if not polygon
  }
  const outerRing = rings[0] || [];  // First outer ring
  if (!Array.isArray(outerRing) || outerRing.length < 3) return [0, 0];  // Invalid
  const minX = Math.min(...outerRing.map(p => p[0]));
  const maxX = Math.max(...outerRing.map(p => p[0]));
  const minY = Math.min(...outerRing.map(p => p[1]));
  const maxY = Math.max(...outerRing.map(p => p[1]));
  let point: [number, number];
  let attempts = 0;
  do {
    point = [minX + Math.random() * (maxX - minX), minY + Math.random() * (maxY - minY)];
    attempts++;
    if (attempts > 100) return [minX, minY];  // Fallback if stuck
  } while (!pointInPolygon(point, outerRing));
  return point;
};

const getEChartsInstance = () => {  // ДОБАВЛЕНО: Выносим функцию как const для внутреннего использования
  return chartRef.value?.$el ? echarts.getInstanceByDom(chartRef.value.$el) : null;
};

defineExpose({
  getEChartsInstance  // ИЗМЕНЕНО: Теперь экспортируем ссылку на const (без повторного определения)
});

const props = defineProps<{
  filters: { from?: string | null; to?: string | null; metric?: 'count' | 'avg_duration' };
}>();

const emit = defineEmits(['selection-updated']);

const getBaseMapOption = (mapName: string, isRegion: boolean = false) => ({
  geo: {
    map: mapName,
    roam: true,
    zoom: isRegion ? 50.0 : 1.0,
    center: [100, 60],
    scaleLimit: { min: 0.8, max: 8 },
    boundingCoords: isRegion ? null : [[30, 40], [180, 82]],
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
        areaColor: '#4f46e5',
        borderColor: '#ffd54f',
        borderWidth: 2,
      },
      label: {
        show: true,
        color: '#e0e0e0',
        fontSize: 14,
        fontWeight: '500',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: [4, 8],
        borderRadius: 4
      }
    },
    animation: false
  },
  tooltip: {
    trigger: 'item',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    textStyle: { color: '#ffc107' },
    formatter: '{b}: <strong>{c}</strong> {seriesName}'
  },
  series: [{
    name: 'Активность полетов',
    type: 'map',
    geoIndex: 0,
    data: [],  // Empty initial for click to work
    emphasis: {
      focus: 'self',
      itemStyle: { areaColor: '#555555' },
      label: { show: true }
    }
  }]
});

const handleChartClick = (params: any) => {
  if (params.seriesType === 'map') {
    if (currentView.value === 'country') {
      selectedRegion.value = params.name;
      selectedDistrict.value = null;
      currentView.value = 'region';
      loadRegionMap();
      emit('selection-updated', { region: params.name, district: null });
    } else if (currentView.value === 'region') {
      // selectedDistrict.value = params.name;
      // emit('selection-updated', { region: selectedRegion.value, district: params.name });
      console.log('Клик по району. Drill-down отключен.');
    }
  }
  console.log('Chart click params:', params);
};

const initMap = async () => {
  try {
    const geoResponse = await axios.get('/maps/Russia.geojson');
    echarts.registerMap('Russia', geoResponse.data);
    option.value = getBaseMapOption('Russia');
    isMapReady.value = true;
    await fetchFlightData();
  } catch (error) {
    console.error('Критическая ошибка: не удалось загрузить GeoJSON карты.', error);
    errorMessage.value = 'Не удалось загрузить файл карты. Проверьте путь и доступность файла.';
  }
};

const fetchFlightData = async () => {
  isLoading.value = true;
  try {
    let flightData = [];
    let apiEndpoint = '/api/regions/flights';
    let params = {
      from: props.filters.from || '2025-01-01',
      to: props.filters.to || '2025-12-31',
      metric: props.filters.metric || 'count',
    };

    if (currentView.value === 'region' && selectedRegion.value) {
      apiEndpoint = '/api/districts/flights';
      params = { ...params, region: selectedRegion.value };
      console.warn(`Заглушка для районов в ${selectedRegion.value}`);
      const geoDataFromCache = cachedGeo.value[selectedRegion.value] || { features: [] };
      const districtNames = geoDataFromCache.features.map((f: any) => f.properties.district || 'Unknown');
      flightData = districtNames.map(name => ({ name, value: Math.floor(Math.random() * 1000) }));
    } else {
      const response = await api.get(apiEndpoint, { params });
      flightData = response.data;
    }

    console.log('Flight data received:', flightData);

    // ИЗМЕНЕНО: Рекомендация - используйте getEChartsInstance() для consistency
    const chartInstance = getEChartsInstance();  // ДОБАВЛЕНО: Вместо chartRef.value
    if (chartInstance && flightData.length > 0) {
      const maxValue = Math.max(...flightData.map((item: { value: number }) => item.value));
      const seriesName = currentView.value === 'region' ? 'полетов в районе' : 'полетов';

      chartInstance.setOption({
        tooltip: { formatter: `{b}: <strong>{c}</strong> ${seriesName}` },
        visualMap: {
          min: 0,
          max: maxValue,
          left: 'left',
          top: 'top',
          orient: 'horizontal',
          text: ['Макс.', 'Мин.'],
          inRange: { color: ['#1a1a1a', '#ffc107'] },
          calculable: true,
          textStyle: { color: '#e0e0e0' }
        },
        series: [{
          name: 'Активность полетов',
          type: 'map',
          geoIndex: 0,
          data: flightData,
          emphasis: {
            focus: 'self',
            itemStyle: { areaColor: '#555555' },
            label: { show: true }
          }
        }]
      });
    } else if (chartInstance) {
      chartInstance.setOption(getBaseMapOption(
        currentView.value === 'region' ? selectedRegion.value || 'Russia' : 'Russia',
        currentView.value === 'region'
      ));
    }
  } catch (error: any) {
    errorMessage.value = error.response?.data?.error || 'Неизвестная ошибка API.';
    console.error('Ошибка при загрузке данных о полётах:', error);
  } finally {
    isLoading.value = false;
  }
};

const loadRegionMap = async () => {
  if (!selectedRegion.value) return;
  try {
    const geoResponse = await axios.get(`/maps/districts/${selectedRegion.value}.geojson`);
    const geoData = geoResponse.data;
    geoData.features.forEach((f: any) => {
      f.properties.name = f.properties.district || 'Unknown';
    });
    echarts.registerMap(selectedRegion.value, geoData);
    cachedGeo.value[selectedRegion.value] = geoData;
    const districtNames = geoData.features.map((f: any) => f.properties.name || 'Unknown');
    const calculatedCenter = calculateCentroid(geoData);    const heatData = [];
    geoData.features.forEach((feature: any) => {
      try {
        const polygon = feature.geometry;
        for (let i = 0; i < 50; i++) {  // ДОБАВЛЕНО: Увеличьте до 100-200 для плотности, если нужно
          const point = randomPointInPolygon(polygon);
          heatData.push([point[0], point[1], 1]);  // Value 1-6 for gradient
        }
      } catch (e) {
        console.warn(`Skip heatmap for district ${feature.properties.district}: ${e.message}`);
      }
    });
    console.log('Heatmap points generated:', heatData.length);

    // ДОБАВЛЕНО: Получаем экземпляр ECharts для consistency с событиями
    const chartInstance = getEChartsInstance();
    if (!chartInstance) {
      throw new Error('ECharts instance not found');
    }

    // ИЗМЕНЕНО: Используем chartInstance.setOption вместо chartRef.value.setOption
    chartInstance.setOption({
      geo: {
        map: selectedRegion.value,
        roam: true,
        zoom: 30,
        center: calculatedCenter,
        scaleLimit: { min: 10, max: 200 },  // Изменено: min 10, max 200 для дистрикт
        itemStyle: {
          areaColor: 'transparent',
          borderColor: '#ffffff',
          borderWidth: 2
        },
        select: {
        itemStyle: {
          areaColor: '#transparent',
          borderColor: '#ffd54f',
          borderWidth: 2,
        },},
        label: {
          show: false,
          color: '#000000',
          fontSize: 14,
          backgroundColor: '#ffffff',
          padding: [2, 4]
        },
        tooltip: {
          trigger: 'item',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          textStyle: { color: '#ffc107' },
          formatter: '{b}'
        },

      },
      visualMap: {
        show: false,
        min: 0,
        max: 5,
        inRange: {
          color: [
            'rgba(33,102,172,0)',
            'rgb(103,169,207)',
            'rgb(209,229,240)',
            'rgb(253,219,199)',
            'rgb(239,138,98)',
            'rgb(178,24,43)'
          ]
        }
      },
      series: [
        {
          id: 'heatmapLayer',
          type: 'heatmap',
          coordinateSystem: 'geo',
          data: heatData,
          blurSize: 3,
          pointSize: 1.5,
          itemStyle: { opacity: 1 }
        },
        {  // Borders layer, transparent no color
          id: 'bordersLayer',
          name: 'Активность полетов',
          type: 'map',
          geoIndex: 0,
          data: districtNames.map(name => ({ name })),
          itemStyle: {
            areaColor: 'transparent',
            borderColor: '#ffffff',
            borderWidth: 2
          },
          emphasis: {
            focus: 'self',
            itemStyle: { areaColor: 'transparent', borderColor: '#ffc107', borderWidth: 2 },
            label: { show: true }
          }
        }
      ]
    });

    // ДОБАВЛЕНО: Dynamic zoom effect из старой версии
    chartInstance.on('georoam', () => {
      const debouncedUpdate = debounce(() => {
      const option = chartInstance.getOption();
      if (!option || !option.geo || !option.geo[0]) return;
      const zoom = option.geo[0].zoom as number;
      const heatmapOpacity = 1;
      const heatmapPointSize = 0.05 * zoom;
      const heatmapBlurSize = 0.1 * zoom;
      chartInstance.setOption({
          series: [
            {
              id: 'heatmapLayer', // <--- УКАЖИ ID
              itemStyle: { opacity: heatmapOpacity },
              pointSize: heatmapPointSize,
              blurSize: heatmapBlurSize
            }
            // Серию 'bordersLayer' можно не трогать, она не обновляется
          ]
        });
      }, 200);
      chartInstance.on('georoam', debouncedUpdate);
    });



  } catch (error) {
    console.error(`Ошибка загрузки GeoJSON для ${selectedRegion.value}:`, error);
    errorMessage.value = `Не удалось загрузить карту для ${selectedRegion.value}.`;
  }
};

const refreshMap = () => {
  // ИЗМЕНЕНО: Используйте getEChartsInstance() вместо chartRef.value
  const chartInstance = getEChartsInstance();
  if (chartInstance) {
    const mapName = currentView.value === 'region' ? selectedRegion.value || 'Russia' : 'Russia';
    const isRegionLocal = currentView.value === 'region';
    const geoDataFromCache = cachedGeo.value[mapName] || { features: [] };
    let newCenter: [number, number] = [95, 65];
    if (isRegionLocal) {
      const calculated = calculateCentroid(geoDataFromCache);
      newCenter = (isNaN(calculated[0]) || isNaN(calculated[1])) ? [95, 65] : calculated;
    }
    chartInstance.setOption({
      geo: {
        zoom: isRegionLocal ? 50.0 : 1.2,
        center: newCenter
      }
    });
    fetchFlightData();
  }
};

const goBack = () => {
  errorMessage.value = '';
  currentView.value = 'country';

  selectedDistrict.value = null;
  if (chartRef.value) {
    chartRef.value.setOption(getBaseMapOption('Russia'));
    fetchFlightData();
  }
};

onMounted(() => {
  initMap();
});
</script>

<style scoped>
/* Ваш текущий стиль без изменений — он логичен и соответствует ТЗ по UX (тёмная тема, responsive) */
.chart-wrapper {
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
  margin: 20px 0;
}

.map-container {
  width: 100%;
  height: 100%;
}

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
  padding: 8px 16px;
  font-size: 0.8rem;
  font-weight: bold;
  color: #000000;
  background: linear-gradient(to right, #ffc107, #ff9800);
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
  box-shadow: 0 8px 16px rgba(255, 193, 7, 0.4);
}

.refresh-icon {
  width: 16px;
  height: 16px;
  color: #000000;
}

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

.back-button {
  position: absolute;
  top: 20px;
  left: 20px;
  padding: 8px 16px;
  font-size: 0.8rem;
  font-weight: bold;
  color: #000000;
  background: linear-gradient(to right, #ff9800, #ffc107);
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

.back-button:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(255, 152, 0, 0.4);
}

.back-icon {
  width: 16px;
  height: 16px;
  color: #000000;
}
</style>
