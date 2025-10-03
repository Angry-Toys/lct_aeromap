<template>
  <div class="dashboard-layout">
    <AppHeader @upload-clicked="isModalVisible = true" />

    <main class="dashboard-content">
      <DashboardFilters @filters-updated="handleFiltersUpdate" />

      <div class="dashboard-grid">
        <div class="main-column">
          <MapOverview :key="componentKey" :filters="activeFilters" />

          <div class="charts-row">
            <TopRegionsChart :key="componentKey + 1" :filters="activeFilters" />
            <HourlyActivityChart :data="metrics.hourlyDistribution" :is-loading="isLoadingMetrics" />
          </div>
        </div>

        <div class="sidebar-column">
          <MetricCard title="Всего полётов" :value="metrics.totalFlights" :is-loading="isLoadingMetrics" />
          <MetricCard title="Средняя длит." :value="metrics.avgDuration" :is-loading="isLoadingMetrics" unit="мин." />
          <MetricCard title="Рост за месяц" :value="metrics.growthPercent" :is-loading="isLoadingMetrics" unit="%" />
          <MetricCard title="Пик. нагрузка (в час)" :value="metrics.peakLoad" :is-loading="isLoadingMetrics" />
          <MetricCard title="Flight Density (средн.)" :value="metrics.flightDensity" :is-loading="isLoadingMetrics" unit="/1k км²" />
          <MetricCard title="Нулевые дни (сумм.)" :value="metrics.zeroDays" :is-loading="isLoadingMetrics" />
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
import MapOverview from '../components/charts/MapOverview.vue';
import TopRegionsChart from '../components/charts/TopRegionsChart.vue';
import HourlyActivityChart from '../components/charts/HourlyActivityChart.vue';
import MetricCard from '../components/ui/MetricCard.vue';
import UploadModal from '../components/ui/UploadModal.vue';
import UploadStatus from '../components/ui/UploadStatus.vue';

// Определяем тип для задачи загрузки, чтобы TypeScript понимал структуру
interface UploadTask {
  id: number;
  file: File;
  progress: number;
  status: 'uploading' | 'success' | 'error' | 'timeout';
}

// --- СОСТОЯНИЕ КОМПОНЕНТА ---

const isModalVisible = ref(false);
const isLoadingMetrics = ref(true);
const componentKey = ref(0); // Ключ для принудительного обновления дочерних компонентов

// Единый объект с фильтрами, который будет обновляться компонентом DashboardFilters
const activeFilters = ref({
  year: '2025',
  month: null,
  metric: 'count',
});

// Реактивный объект для хранения всех метрик сайдбара
const metrics = reactive({
  totalFlights: null, avgDuration: null, growthPercent: null,
  peakLoad: null, flightDensity: null, zeroDays: null,
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
        year: activeFilters.value.year,
        month: activeFilters.value.month
      }
    });

    let regionsData = response.data;

    // РЕШЕНИЕ ОШИБКИ: Проверяем, является ли ответ массивом. Если нет - создаем массив.
    if (!Array.isArray(regionsData)) {
      regionsData = [regionsData];
    }

    if (regionsData && regionsData.length > 0) {
      // Агрегируем общую статистику для сайдбара, суммируя данные по всем регионам
      const totals = regionsData.reduce((acc: any, region: any) => {
        acc.totalFlights += region.flight_count || 0;
        acc.totalDuration += region.total_duration_min || 0;
        acc.peakLoad = Math.max(acc.peakLoad, region.peak_load || 0);
        acc.zeroDays += region.zero_days || 0;
        if (region.growth_percent != null) acc.growthPercentList.push(region.growth_percent);
        if (region.flight_density != null) acc.flightDensityList.push(region.flight_density);
        return acc;
      }, {
        totalFlights: 0, totalDuration: 0, peakLoad: 0, zeroDays: 0,
        growthPercentList: [], flightDensityList: []
      });

      // Присваиваем вычисленные значения
      metrics.totalFlights = totals.totalFlights;
      metrics.avgDuration = totals.totalFlights > 0 ? Math.round(totals.totalDuration / totals.totalFlights) : 0;
      metrics.peakLoad = totals.peakLoad;
      metrics.zeroDays = totals.zeroDays;
      metrics.growthPercent = totals.growthPercentList.length > 0 ? parseFloat((totals.growthPercentList.reduce((a: number, b: number) => a + b, 0) / totals.growthPercentList.length).toFixed(1)) : 0;
      metrics.flightDensity = totals.flightDensityList.length > 0 ? parseFloat((totals.flightDensityList.reduce((a: number, b: number) => a + b, 0) / totals.flightDensityList.length).toFixed(2)) : 0;
      metrics.hourlyDistribution = regionsData[0].hourly_distribution || null;
    }

  } catch (error) {
    console.error("Не удалось загрузить метрики для сайдбара:", error);
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
    status: 'uploading'
  };
  uploadTasks.value.push(newTask);
  executeUpload(newTask);
};

const executeUpload = async (task: UploadTask) => {
  const formData = new FormData();
  formData.append('file', task.file);
  try {
    await axios.post('http://localhost:5000/upload', formData, {
      timeout: 30000, // Таймаут 30 секунд
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          const taskInArray = uploadTasks.value.find(t => t.id === task.id);
          if (taskInArray) {
            taskInArray.progress = percentCompleted;
          }
        }
      }
    });
    const taskInArray = uploadTasks.value.find(t => t.id === task.id);
    if (taskInArray) {
      taskInArray.status = 'success';
      handleFiltersUpdate(activeFilters.value); // Обновляем все данные на дашборде
    }
  } catch (error) {
    const taskInArray = uploadTasks.value.find(t => t.id === task.id);
    if (taskInArray) {
      if (axios.isAxiosError(error) && error.code === 'ECONNABORTED') {
        taskInArray.status = 'timeout';
      } else {
        taskInArray.status = 'error';
      }
    }
    console.error("Ошибка при загрузке файла:", error);
  }
};


// --- ОБРАБОТЧИКИ СОБЫТИЙ ---

const handleFiltersUpdate = (filters: any) => {
  activeFilters.value = filters;
  fetchDataForSidebar();
  componentKey.value++;
};

const handleModalClose = () => {
  isModalVisible.value = false;
};

// --- ХУК ЖИЗНЕННОГО ЦИКЛА ---
onMounted(fetchDataForSidebar);

</script>

<style scoped>
.dashboard-layout {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #0a1929;
}
.dashboard-content {
  flex: 1; /* Занимает всё оставшееся место после хедера */
  padding: 24px;
  overflow-y: auto; /* Включаем скролл, если контент не помещается */
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-height: 0; /* Важное правило для flex-контейнеров */
}
.dashboard-grid {
  flex: 1; /* Сетка растягивается на всю высоту */
  display: grid;
  gap: 24px;
  grid-template-columns: 3fr 1fr; /* 3 части для основного контента, 1 для сайдбара */
}
.main-column {
  display: grid;
  grid-template-rows: 2fr 1fr; /* 2 части для карты, 1 для ряда графиков */
  gap: 24px;
  min-height: 0;
}
.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  min-width: 0; /* Предотвращает сжатие контента */
}
.sidebar-column {
  display: grid;
  grid-auto-rows: min-content; /* Ячейки занимают высоту своего контента */
  gap: 24px;
}
</style>
