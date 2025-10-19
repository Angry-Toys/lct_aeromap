<template>
  <div ref="chartContainer" class="map"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue';
import * as echarts from 'echarts';
import type { ECharts, EChartsOption } from 'echarts';
import axios from 'axios';
import * as turf from '@turf/turf';

interface FlightItem {
  name: string;
  value: number;
}

type ExtendedFeature = GeoJSON.Feature & { properties: { [key: string]: any } };

// --- Component State ---
const chartContainer = ref<HTMLElement | null>(null);
let chart: ECharts | null = null;

// --- ECharts Integration ---
const initialize = async () => {
  if (!chartContainer.value) {
    console.error('Chart container not found.');
    return;
  }

  try {
    chart = echarts.init(chartContainer.value);

    const [regionsResponse, geoJsonResponse] = await Promise.all([
      axios.get<FlightItem[]>('/api/api/regions/flights?from=2025-01-01&to=2025-12-31'),
      fetch('/maps/Russia.geojson').then(res => res.json() as Promise<GeoJSON.FeatureCollection>)
    ]);

    const flightData = regionsResponse.data;
    const geoJsonData = geoJsonResponse;

    if (!Array.isArray(flightData) || !geoJsonData?.features) {
      throw new Error('Invalid data format.');
    }

    geoJsonData.features = geoJsonData.features.filter((feature: ExtendedFeature) =>
      flightData.some(item => item.name === feature.properties.name_ru)
    );

    // Объединяем полигоны в MultiPolygon для фильтра точек
    const allPolygons = geoJsonData.features.map(f => f.geometry.coordinates as number[][][]);
    const multiPolygon = turf.multiPolygon(allPolygons);

    // Генерация точек внутри границ
    const overallBbox = turf.bbox(multiPolygon);
    let heatmapData = turf.randomPoint(15000, { bbox: overallBbox });
    heatmapData.features = heatmapData.features.filter(point =>
      turf.booleanPointInPolygon(point, multiPolygon)
    ).slice(0, 5000);

    const heatmapPoints = heatmapData.features.map(f => [
      f.geometry.coordinates[0],
      f.geometry.coordinates[1],
      2
    ]);

    echarts.registerMap('russia', geoJsonData);

    const option: EChartsOption = {
      geo: {
        map: 'russia',
        roam: true,
        itemStyle: {
          normal: {
            areaColor: 'transparent',
            borderColor: '#ffffff',
            borderWidth: 2
          }
        },
        label: {
          show: true,
          formatter: '{name_ru}',
          color: '#000000',
          fontSize: 14,
          backgroundColor: '#ffffff',
          padding: [2, 4]
        }
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
          type: 'heatmap',
          coordinateSystem: 'geo',
          data: heatmapPoints,
          blurSize: 20,
          pointSize: 20,
          itemStyle: { opacity: 1 }
        },
        {
          type: 'scatter',
          coordinateSystem: 'geo',
          data: heatmapPoints,
          symbolSize: (val) => val[2] * 2 + 3, // Мелкие точки: 3–13 px
          itemStyle: {
            opacity: 0,
            color: [
              'interpolate',
              ['linear'],
              ['get', 2],
              1, 'rgba(33,102,172,0)',
              2, 'rgb(103,169,207)',
              3, 'rgb(209,229,240)',
              4, 'rgb(253,219,199)',
              5, 'rgb(239,138,98)',
              6, 'rgb(178,24,43)'
            ]
          }
        }
      ]
    };

    chart.setOption(option);

    // Эффект зума: Heatmap гаснет, точки появляются, размер heatmap фиксируется при малом зуме
    chart.on('georoam', () => {
      const zoom = chart.getOption().geo[0].zoom as number;
      const heatmapOpacity = 1;
      const scatterOpacity = 0;
      const heatmapPointSize = Math.max(1, 1 * (1* zoom)); // Уменьшаем размер при малом зуме (удалении)
      const heatmapBlurSize = Math.max(1, 5 * (1 * zoom)); // Аналогично для размытия
      chart.setOption({
        series: [
          {
            type: 'heatmap',
            itemStyle: { opacity: heatmapOpacity },
            pointSize: heatmapPointSize,
            blurSize: heatmapBlurSize
          },
          { type: 'scatter', itemStyle: { opacity: scatterOpacity } }
        ]
      });
    });

  } catch (error: any) {
    console.error('Error during chart setup:', error);
    alert(error.message || 'Failed to load map data.');
  }
};

onMounted(async () => {
  if (typeof window === 'undefined') return;
  await nextTick();
  initialize();
});

onUnmounted(() => {
  if (chart) {
    chart.dispose();
    chart = null;
  }
});
</script>

<style scoped>
.map {
  width: 100%;
  height: 600px;
}
</style>
