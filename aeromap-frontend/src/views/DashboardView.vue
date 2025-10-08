<template>
  <div class="dashboard-layout">
    <AppHeader @upload-clicked="isModalVisible = true" />

    <main class="dashboard-content">
      <DashboardFilters @filters-updated="handleFiltersUpdate" />
      <SelectedParams :filters="activeFilters" />

      <!-- Новый контейнер для горизонтальных карточек метрик -->
      <div class="metrics-row">
        <MetricCard title="Всего полётов" :value="metrics.totalFlights" :is-loading="isLoadingMetrics" />
        <MetricCard title="Средняя длит." :value="metrics.avgDuration" :is-loading="isLoadingMetrics" unit="мин." />
        <MetricCard title="Рост за месяц" :value="metrics.growthPercent" :is-loading="isLoadingMetrics" unit="%" />
        <MetricCard title="Пик. нагрузка (в час)" :value="metrics.peakLoad" :is-loading="isLoadingMetrics" />
        <MetricCard title="Flight Density (средн.)" :value="metrics.flightDensity" :is-loading="isLoadingMetrics" unit="/1k км²" />
        <MetricCard title="Нулевые дни (сумм.)" :value="metrics.zeroDays" :is-loading="isLoadingMetrics" />
      </div>

      <div class="dashboard-grid">
        <div class="main-column">
          <MapOverview :key="componentKey" :filters="activeFilters" />
          <div class="charts-row">
            <TopRegionsChart :key="componentKey + 1" :filters="activeFilters" />
            <HourlyActivityChart :data="metrics.hourlyDistribution" :is-loading="isLoadingMetrics" />
          </div>
        </div>
      </div>
    </main>

    <UploadModal
      v-if="isModalVisible"
      @close="isModalVisible = false"
      @start-upload="handleStartUpload"
    />

    <UploadStatus :tasks="uploadTasks" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue';
import axios from 'axios';

// Импорты всех компонентов
import AppHeader from '../components/layout/AppHeader.vue';
import DashboardFilters from '../components/layout/DashboardFilters.vue';
import SelectedParams from '../components/ui/SelectedParams.vue';
import MapOverview from '../components/charts/MapOverview.vue';
import TopRegionsChart from '../components/charts/TopRegionsChart.vue';
import HourlyActivityChart from '../components/charts/HourlyActivityChart.vue';
import MetricCard from '../components/ui/MetricCard.vue';
import UploadModal from '../components/ui/UploadModal.vue';
import UploadStatus from '../components/ui/UploadStatus.vue';

// Определяем тип для задачи загрузки
interface UploadTask {
  id: number;
  file: File;
  progress: number;
  status: 'uploading' | 'success' | 'error' | 'timeout';
}

// Тип для фильтров
interface Filters {
  from?: string | null;
  to?: string | null;
  metric?: 'count' | 'avg_duration';
}

// --- СОСТОЯНИЕ КОМПОНЕНТА ---

const isModalVisible = ref(false);
const isLoadingMetrics = ref(true);
const componentKey = ref(0);

// Единый объект с фильтрами
const activeFilters = ref<Filters>({
  from: '2025-01-01',
  to: '2025-12-31',
  metric: 'count',
});

// Реактивный объект для хранения всех метрик
const metrics = reactive({
  totalFlights: null,
  avgDuration: null,
  growthPercent: null,
  peakLoad: null,
  flightDensity: null,
  zeroDays: null,
  hourlyDistribution: null,
});

// Массив для хранения всех фоновых задач на загрузку
const uploadTasks = ref<UploadTask[]>([]);

// --- ЛОГИКА РАБОТЫ С ДАННЫМИ ---

const fetchDataForSidebar = async () => {
  isLoadingMetrics.value = true;
  try {
    const response = await axios.get('http://localhost:5000/metrics', {
      params: {
        year: activeFilters.value.from?.split('-')[0],
        month: activeFilters.value.from?.split('-')[1],
      },
    });

    let regionsData = response.data;

    if (!Array.isArray(regionsData)) {
      regionsData = [regionsData];
    }

    if (regionsData && regionsData.length > 0) {
      const totals = regionsData.reduce(
        (acc: any, region: any) => {
          acc.totalFlights += region.flight_count || 0;
          acc.totalDuration += region.total_duration_min || 0;
          acc.peakLoad = Math.max(acc.peakLoad, region.peak_load || 0);
          acc.growthPercentList.push(region.growth_percent || 0);
          acc.flightDensityList.push(region.flight_density || 0);
          return acc;
        },
        {
          totalFlights: 0,
          totalDuration: 0,
          peakLoad: 0,
          growthPercentList: [],
          flightDensityList: [],
          zeroDays: 0,
        }
      );

      metrics.totalFlights = totals.totalFlights;
      metrics.avgDuration = totals.totalFlights
        ? parseFloat((totals.totalDuration / totals.totalFlights).toFixed(1))
        : 0;
      metrics.peakLoad = totals.peakLoad;
      metrics.zeroDays = regionsData.reduce(
        (acc: number, region: any) => acc + (region.zero_days || 0),
        0
      );
      metrics.growthPercent =
        totals.growthPercentList.length > 0
          ? parseFloat(
              (
                totals.growthPercentList.reduce((a: number, b: number) => a + b, 0) /
                totals.growthPercentList.length
              ).toFixed(1)
            )
          : 0;
      metrics.flightDensity =
        totals.flightDensityList.length > 0
          ? parseFloat(
              (
                totals.flightDensityList.reduce((a: number, b: number) => a + b, 0) /
                totals.flightDensityList.length
              ).toFixed(2)
            )
          : 0;
      metrics.hourlyDistribution = regionsData[0].hourly_distribution || null;
    }
  } catch (error) {
    console.error('Не удалось загрузить метрики для сайдбара:', error);
  } finally {
    isLoadingMetrics.value = false;
  }
};

// --- ЛОГИКА ЗАГРУЗКИ ФАЙЛОВ ---

const handleStartUpload = (file: File) => {
  const newTask: UploadTask = {
    id: Date.now(),
    file,
    progress: 0,
    status: 'uploading',
  };
  uploadTasks.value.push(newTask);
  executeUpload(newTask);
};

const executeUpload = async (task: UploadTask) => {
  const formData = new FormData();
  formData.append('file', task.file);
  try {
    await axios.post('http://localhost:5000/upload', formData, {
      timeout: 30000,
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          const taskInArray = uploadTasks.value.find((t) => t.id === task.id);
          if (taskInArray) {
            taskInArray.progress = percentCompleted;
          }
        }
      },
    });
    const taskInArray = uploadTasks.value.find((t) => t.id === task.id);
    if (taskInArray) {
      taskInArray.status = 'success';
      handleFiltersUpdate(activeFilters.value);
    }
  } catch (error) {
    const taskInArray = uploadTasks.value.find((t) => t.id === task.id);
    if (taskInArray) {
      if (axios.isAxiosError(error) && error.code === 'ECONNABORTED') {
        taskInArray.status = 'timeout';
      } else {
        taskInArray.status = 'error';
      }
    }
    console.error('Ошибка при загрузке файла:', error);
  }
};

// --- ОБРАБОТЧИКИ СОБЫТИЙ ---

const handleFiltersUpdate = (filters: Filters) => {
  console.log('Received filters in DashboardView:', filters);
  activeFilters.value = {
    from: filters.from || '2025-01-01',
    to: filters.to || '2025-12-31',
    metric: filters.metric || 'count',
  };
  fetchDataForSidebar();
  componentKey.value++;
};

// --- ХУК ЖИЗНЕННОГО ЦИКЛА ---
onMounted(fetchDataForSidebar);
</script>

<style scoped>
.dashboard-layout {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #000000;
}

.dashboard-content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-height: 0;
}

.dashboard-grid {
  flex: 1;
  display: grid;
  gap: 32px; /* Увеличено с 24px до 32px для большего отступа от метрик */
  grid-template-columns: 1fr;
}

.main-column {
  display: grid;
  grid-template-rows: 2fr 1fr;
  gap: 24px;
  min-height: 0;
  width: 100%;
}

.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  min-width: 0;
}

.metrics-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  animation: fadeIn 0.3s ease-in; /* Добавлена анимация */
}

/* Стили для MetricCard */
:deep(.metric-card) {
  flex: 1;
  min-width: 200px;
  box-sizing: border-box;
}

/* Анимация появления */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Адаптивность */
@media (max-width: 1200px) {
  .metrics-row {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  }
}

@media (max-width: 768px) {
  .metrics-row {
    grid-template-columns: 1fr;
  }
  .charts-row {
    grid-template-columns: 1fr;
  }
}
</style>
