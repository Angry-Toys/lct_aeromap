<template>
  <div class="dashboard-layout">
    <AppHeader @upload-clicked="isModalVisible = true" />

    <main class="dashboard-content">
      <DashboardFilters @filters-updated="handleFiltersUpdate" />
      <SelectedParams :filters="activeFilters" :selectedRegion="selectedRegion" :missingMonths="missingMonths" />

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
          <MapOverview
            :key="componentKey"
            :filters="activeFilters"
            @region-selected="handleRegionSelected"
          />
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
const selectedRegion = ref<string>('Russian Federation');
const missingMonths = ref<string[]>([]); // Для хранения 'YYYY-MM' без данных

const handleRegionSelected = (region: string) => {
  console.log('Region selected:', region);
  selectedRegion.value = region;
  debouncedFetch();
};

// Единый объект с фильтрами
const activeFilters = ref<Filters>({
  from: '2025-01-01',
  to: '2025-12-31',
  metric: 'count',
});

// Реактивный объект для хранения всех метрик
const metrics = reactive({
  totalFlights: null as number | null,
  avgDuration: null as number | null,
  growthPercent: null as number | null,
  peakLoad: null as number | null,
  flightDensity: null as number | null,
  zeroDays: null as number | null,
  hourlyDistribution: null as { hour: number; count: number }[] | null,
});

// Массив для хранения всех фоновых задач на загрузку
const uploadTasks = ref<UploadTask[]>([]);

// Debounce для fetchDataForSidebar (задержка 300мс)
let debounceTimer: number | null = null;
const debouncedFetch = () => {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    fetchDataForSidebar();
  }, 300);
};

// --- ЛОГИКА РАБОТЫ С ДАННЫМИ ---

// Функция для получения списка месяцев в периоде
const getMonthsInRange = (from: string, to: string) => {
  const start = new Date(from);
  const end = new Date(to);
  const months: { year: number; month: string }[] = [];
  const current = new Date(start.getFullYear(), start.getMonth(), 1);
  while (current <= end) {
    months.push({
      year: current.getFullYear(),
      month: String(current.getMonth() + 1).padStart(2, '0'),
    });
    current.setMonth(current.getMonth() + 1);
  }
  return months;
};

const fetchDataForSidebar = async () => {
  isLoadingMetrics.value = true;
  try {
    const from = activeFilters.value.from || '2025-01-01';
    const to = activeFilters.value.to || '2025-12-31';
    const selected = selectedRegion.value;
    const isAllRussia = selected === 'Russian Federation';

    console.log('🚀 Загрузка метрик:', { region: selected, period: { from, to } });

    // Получаем список месяцев
    const months = getMonthsInRange(from, to);
    missingMonths.value = [];

    // Параллельные вызовы /metrics для каждого месяца
    const promises = months.map(async ({ year, month }, index) => {
      const params: any = { year: String(year), month };
      if (!isAllRussia) {
        params.region = selected;
      }
      const url = `http://localhost:5000/metrics?${new URLSearchParams(params).toString()}`;
      console.log(`📡 Запрос ${index + 1}/${months.length}:`, { url, params });

      try {
        const response = await axios.get('http://localhost:5000/metrics', { params });
        console.log(`✅ Ответ для ${url}:`, response.data);
        let data = response.data;
        if (!Array.isArray(data)) {
          data = [data];
        }
        if (data.length === 0) {
          missingMonths.value.push(`${year}-${month}`);
        }
        return data;
      } catch (error: any) {
        console.error(`❌ Ошибка для ${url}:`, {
          params,
          error: error.message,
          response: error.response?.data || 'Нет данных ответа',
        });
        missingMonths.value.push(`${year}-${month}`);
        return [];
      }
    });

    const monthResponses = await Promise.all(promises);
    const allRegionsData = monthResponses.flat();

    if (allRegionsData.length === 0) {
      console.warn('⚠️ Нет данных от API /metrics. Устанавливаем метрики в 0.');
      metrics.totalFlights = 0;
      metrics.avgDuration = 0;
      metrics.peakLoad = 0;
      metrics.flightDensity = 0;
      metrics.zeroDays = 0;
      metrics.growthPercent = 0;
      metrics.hourlyDistribution = null;
      return;
    }

    // Агрегация
    const totals = {
      totalFlights: 0,
      totalDuration: 0,
      peakLoad: 0,
      zeroDays: 0,
      growthPercentList: [] as number[],
      flightDensityList: [] as number[],
      hourlySums: Array(24).fill(0),
      hasHourlyData: false,
      hasPeakLoad: false,
      hasFlightDensity: false,
    };

    allRegionsData.forEach((region: any) => {
      totals.totalFlights += region.flight_count || 0;
      totals.totalDuration += region.total_duration_min || 0;
      totals.peakLoad = Math.max(totals.peakLoad, region.peak_load_hourly || 0);
      totals.zeroDays += region.zero_days || 0;
      totals.growthPercentList.push(region.growth_percent || 0);
      totals.flightDensityList.push(region.flight_density || 0);

      // Отмечаем наличие данных
      if (region.peak_load_hourly || region.peak_load_hourly === 0) {
        totals.hasPeakLoad = true;
      }
      if (region.flight_density || region.flight_density === 0) {
        totals.hasFlightDensity = true;
      }

      // Суммируем hourly_distribution
      const hourly = region.hourly_distribution || [];
      if (hourly.length > 0) {
        totals.hasHourlyData = true;
      }
      hourly.forEach((h: any) => {
        const hour = Math.floor(h.hour);
        if (hour >= 0 && hour < 24) {
          totals.hourlySums[hour] += h.count || 0;
        }
      });
    });

    // Проверка на отсутствие данных
    if (!totals.hasPeakLoad) {
      console.warn('⚠️ peak_load_hourly отсутствует во всех данных');
    }
    if (!totals.hasFlightDensity) {
      console.warn('⚠️ flight_density отсутствует во всех данных');
    }
    if (!totals.hasHourlyData) {
      console.warn('⚠️ hourly_distribution отсутствует или пусто во всех данных');
    }

    // Вычисления
    metrics.totalFlights = totals.totalFlights;
    metrics.avgDuration = totals.totalFlights > 0 ? parseFloat((totals.totalDuration / totals.totalFlights).toFixed(1)) : 0;
    metrics.peakLoad = totals.peakLoad || 0;
    metrics.zeroDays = totals.zeroDays;
    metrics.growthPercent = totals.growthPercentList.length > 0
      ? parseFloat((totals.growthPercentList.reduce((a, b) => a + b, 0) / totals.growthPercentList.length).toFixed(1))
      : 0;
    metrics.flightDensity = totals.flightDensityList.length > 0
      ? parseFloat((totals.flightDensityList.reduce((a, b) => a + b, 0) / totals.flightDensityList.length).toFixed(4))
      : 0;

    // Среднесуточная hourly динамика
    const startDate = new Date(from);
    const endDate = new Date(to);
    const totalDays = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)) + 1;
    const hourlyDistributionTemp = totals.hourlySums.map((sumCount, hour) => ({
      hour,
      count: totalDays > 0 ? Math.round(sumCount / totalDays) : 0,
    }));
    metrics.hourlyDistribution = totals.hourlySums.every(c => c === 0) ? null : hourlyDistributionTemp;

    console.log('🎯 Финальные метрики:', {
      totalFlights: metrics.totalFlights,
      avgDuration: metrics.avgDuration,
      peakLoad: metrics.peakLoad,
      flightDensity: metrics.flightDensity,
      zeroDays: metrics.zeroDays,
      growthPercent: metrics.growthPercent,
      hourlyDistribution: metrics.hourlyDistribution,
    });
  } catch (error) {
    console.error('❌ Критическая ошибка в fetchDataForSidebar:', error);
    metrics.totalFlights = 0;
    metrics.avgDuration = 0;
    metrics.peakLoad = 0;
    metrics.flightDensity = 0;
    metrics.zeroDays = 0;
    metrics.growthPercent = 0;
    metrics.hourlyDistribution = null;
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
  debouncedFetch();
  componentKey.value++;
};

// --- ХУК ЖИЗНЕННОГО ЦИКЛА ---
onMounted(() => fetchDataForSidebar());
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
  padding: 24px 64px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-height: 0;
}

.dashboard-grid {
  flex: 1;
  display: grid;
  gap: 32px;
  grid-template-columns: 1fr;
}

.main-column {
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
  animation: fadeIn 0.3s ease-in;
}

:deep(.metric-card) {
  flex: 1;
  min-width: 200px;
  box-sizing: border-box;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

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
