<template>
  <div class="dashboard-layout">
    <AppHeader @upload-clicked="isModalVisible = true" @export-json="handleExportJson" @generate-png="isExportModalVisible = true"/>

    <main class="dashboard-content">
      <DashboardFilters @filters-updated="handleFiltersUpdate" />
      <SelectedParams :filters="activeFilters" :selectedPath="selectedPath" :missingMonths="missingMonths" />

      <div class="metrics-row" ref="metricsRowRef">
        <MetricCard title="Всего полётов" :value="metrics.totalFlights" :is-loading="isLoadingMetrics" />
        <MetricCard title="Средняя длительность" :value="metrics.avgDuration" :is-loading="isLoadingMetrics" unit="мин." />
        <MetricCard
          title="Рост за период"
          :value="metrics.growthPercent"
          :is-loading="isLoadingMetrics"
          unit="%"
          @card-click="showGrowthModal = true"
        />
        <MetricCard title="Пик. нагрузка (в час)" :value="metrics.peakLoad" :is-loading="isLoadingMetrics" />
        <MetricCard title="Flight Density (средн.)" :value="metrics.flightDensity" :is-loading="isLoadingMetrics" unit="/1k км²" />
        <MetricCard title="Нулевые дни (сумма)" :value="metrics.zeroDays" :is-loading="isLoadingMetrics" />
      </div>

      <div class="dashboard-grid">
        <div class="main-column">
          <MapOverview
            ref="mapOverviewRef"
            :key="componentKey"
            :filters="activeFilters"
            @selection-updated="handleSelectionUpdated"
          />
          <div class="charts-row">
            <TopRegionsChart ref="topRegionsChartRef" :key="componentKey + 1" :filters="activeFilters" />
            <HourlyActivityChart ref="hourlyActivityChartRef" :data="metrics.hourlyDistribution" :is-loading="isLoadingMetrics" />
          </div>
        </div>
      </div>
    </main>
    <ExportModal
        :visible="isExportModalVisible"
        @close="isExportModalVisible = false"
        @generate="handleGenerateReport"
      />

    <UploadModal
      v-if="isModalVisible"
      @close="isModalVisible = false"
      @start-upload="handleStartUpload"
    />
    <GrowthChartModal
      v-if="showGrowthModal"
      :data="growthPercentData"
      @close="showGrowthModal = false"
    />
    <UploadStatus :tasks="uploadTasks" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive, computed  } from 'vue';
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
import GrowthChartModal from '../components/ui/GrowthChartModal.vue';
import ExportModal from '../components/ui/ExportModal.vue';
import html2canvas from 'html2canvas'; // Библиотека для скриншота метрик
import api from '@/utils/api';  // Импорт глобального api с токеном

// Определяем тип для задачи загрузки
interface UploadTask {
  id: number;
  file: File;
  progress: number;
  status: 'uploading' | 'processing' | 'success' | 'error' | 'timeout';
  errorMessage?: string;
}

// Тип для фильтров
interface Filters {
  from?: string | null;
  to?: string | null;
  metric?: 'count' | 'avg_duration';
  customer?: string | null; // <-- ДОБАВЛЕНО
}

// Тип для данных графика роста
interface GrowthData {
  month: string;
  value: number;
}

// --- СОСТОЯНИЕ КОМПОНЕНТА ---
const isModalVisible = ref(false);
const isLoadingMetrics = ref(true);
const componentKey = ref(0);
const selectedRegion = ref<string>('Российская Федерация');
const selectedDistrict = ref<string | null>(null);

const missingMonths = ref<string[]>([]);
const showGrowthModal = ref(false);
const growthPercentData = ref<GrowthData[]>([]);

const selectedPath = computed(() => {
  return selectedDistrict.value ? `${selectedRegion.value} > ${selectedDistrict.value}` : selectedRegion.value;
});

const metricsRowRef = ref<HTMLElement | null>(null);
const mapOverviewRef = ref(null);
const topRegionsChartRef = ref(null);
const hourlyActivityChartRef = ref(null);


const isExportModalVisible = ref(false);
const handleGenerateReport = async (selectedIds: string[]) => {
  if (selectedIds.length === 0) {
    alert("Пожалуйста, выберите хотя бы один элемент для экспорта.");
    return;
  }

  const padding = 80;
  const gap = 40;
  const qualityScale = 2;
  const capturedImages = [];

  const elementRefs = {
    metrics: metricsRowRef,
    map: mapOverviewRef,
    topRegions: topRegionsChartRef,
    hourlyActivity: hourlyActivityChartRef,
  };

  for (const id of selectedIds) {
    const componentRef = elementRefs[id as keyof typeof elementRefs];
    if (componentRef.value) {
      const elementToCapture = (componentRef.value as any).$el || componentRef.value;

      const canvas = await html2canvas(elementToCapture, {
        scale: qualityScale,
        backgroundColor: null, // Оставляем null, так как фон задаем ниже
        useCORS: true,
        logging: false,
        ignoreElements: (element) => element.hasAttribute('data-html2canvas-ignore'),

        // --- РЕШЕНИЕ ПРОБЛЕМЫ С ТУСКЛОСТЬЮ ---
        onclone: (documentClone) => {
          // Находим склонированный элемент метрик и принудительно
          // делаем его фон сплошным черным для корректного рендеринга.
          // Это не влияет на оригинальный элемент на странице.
          if (id === 'metrics') {
            const clonedElement = documentClone.querySelector('.metrics-row');
            if (clonedElement) {
              (clonedElement as HTMLElement).style.backgroundColor = '#000000';
            }
          }
        },
      });

      capturedImages.push({
        id,
        src: canvas.toDataURL('image/png'),
        width: canvas.width,
        height: canvas.height,
      });
    }
  }

  // --- Блок расчета размеров и отрисовки (без изменений) ---
  const metricsImage = capturedImages.find(img => img.id === 'metrics');
  const mapImage = capturedImages.find(img => img.id === 'map');
  const smallCharts = capturedImages.filter(img => !['metrics', 'map'].includes(img.id));
  const smallChartsWidth = smallCharts.reduce((sum, img) => sum + img.width, 0) + (smallCharts.length > 1 ? gap : 0);
  const canvasWidth = padding * 2 + Math.max(metricsImage?.width ?? 0, mapImage?.width ?? 0, smallChartsWidth);
  let canvasHeight = padding;
  if (metricsImage) canvasHeight += metricsImage.height + gap;
  if (mapImage) canvasHeight += mapImage.height + gap;
  if (smallCharts.length > 0) canvasHeight += (smallCharts[0]?.height ?? 0);
  canvasHeight = (canvasHeight - gap) + padding;

  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = canvasWidth;
  finalCanvas.height = canvasHeight;
  const ctx = finalCanvas.getContext('2d')!;

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);

  let currentY = padding;
  const drawCentered = async (image: any) => {
    if (!image) return;
    const img = await loadImage(image.src);
    const x = (canvasWidth - image.width) / 2;
    ctx.drawImage(img, x, currentY, image.width, image.height);
    currentY += image.height + gap;
  };

  await drawCentered(metricsImage);
  await drawCentered(mapImage);

  if (smallCharts.length > 0) {
    let currentX = (canvasWidth - smallChartsWidth) / 2;
    for (const chart of smallCharts) {
      const img = await loadImage(chart.src);
      ctx.drawImage(img, currentX, currentY, chart.width, chart.height);
      currentX += chart.width + gap;
    }
  }

  const link = document.createElement('a');
  link.download = `aeromap-report-${new Date().toISOString().split('T')[0]}.png`;
  link.href = finalCanvas.toDataURL('image/png');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

const loadImage = (src: string): Promise<HTMLImageElement> => {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
};


const handleSelectionUpdated = ({ region, district }: { region: string; district: string | null }) => {
  console.log('Selection updated:', { region, district });
  selectedRegion.value = region;
  selectedDistrict.value = district;
  debouncedFetch();
};


// Единый объект с фильтрами
const activeFilters = ref<Filters>({
  from: '2025-01-01',
  to: '2025-12-31',
  metric: 'count',
  customer: null, // <-- ДОБАВЛЕНО
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

// Функция для получения количества дней в месяце
const daysInMonth = (year: number, month: number) => new Date(year, month, 0).getDate();

const fetchDataForSidebar = async () => {
  // Сброс метрик перед загрузкой
  metrics.totalFlights = null;
  metrics.avgDuration = null;
  metrics.peakLoad = null;
  metrics.flightDensity = null;
  metrics.zeroDays = null;
  metrics.growthPercent = null;
  metrics.hourlyDistribution = null;
  growthPercentData.value = [];
  missingMonths.value = [];
  isLoadingMetrics.value = true;

  // --- 1. СБОР ПАРАМЕТРОВ ---
  const from = activeFilters.value.from || '2025-01-01';
  const to = activeFilters.value.to || '2025-12-31';
  const selected = selectedRegion.value;
  const selectedDistrictLocal = selectedDistrict.value;
  const customer = activeFilters.value.customer;
  const isAllRussia = selected === 'Российская Федерация';
  const apiPath = isAllRussia ? '/api/metrics/customers' : `/api/metrics/customers/${selected}`;

  const params = {
    from,
    to,
    customer: customer || undefined, // undefined, чтобы не отправлять 'null'
  };

  // --- 2. ЛОГ НАЧАЛА ---
  console.group('📡 Запрос метрик: fetchDataForSidebar');
  console.info(`🎯 API Endpoint: ${apiPath}`);
  console.log('➡️ Фильтры:', { region: selected, period: { from, to }, customer: customer || 'Все' });
  console.log('➡️ Параметры запроса (Query Params):', params);

  // --- 3. ЗАГЛУШКИ ---
  if (selectedDistrictLocal) {
    console.warn(`⚠️ Заглушка: Данные по району ${selectedDistrictLocal} не найдены.`);
    isLoadingMetrics.value = false;
    console.groupEnd();
    return;
  }

  if (isAllRussia) {
    console.warn('⚠️ Заглушка: Логика для Russian Federation (агрегация/особый кейс) не реализована. Продолжаю выполнять запрос к API.');
    // Если есть специальная заглушка или сброс для "Вся Россия", он должен быть тут.
  }

  try {
    // --- 4. ВЫЗОВ API ---
    const response = await axios.get(apiPath, { params });
    console.log(`✅ Успешный ответ (HTTP 200). Тип данных до обработки:`, typeof response.data);
    let data = response.data;

    // ⚠️ КОРРЕКЦИЯ: ИСПРАВЛЕНИЕ ОШИБКИ NaN В JSON
    if (typeof data === 'string' && data.length > 0) {
        // 1. Санитаризация: JSON не поддерживает NaN. Заменяем его на null.
        // Используем регулярное выражение /NaN/g, чтобы заменить ВСЕ вхождения.
        const sanitizedString = data.replace(/NaN/g, 'null');

        try {
            data = JSON.parse(sanitizedString);
            console.log('🔄 Успешный принудительный парсинг JSON после санитаризации (NaN -> null).');
        } catch (e) {
            console.error('❌ Ошибка при попытке парсинга строки в JSON (даже после санитаризации):', e);
            // Если парсинг не удался, сбрасываем данные
            data = null;
        }
    }

    let finalMetricsData: any = {};

    // --- 5. ОБРАБОТКА И АГРЕГАЦИЯ ДАННЫХ ---
    if (Array.isArray(data)) {
        // Кейс: выбран 'Все' заказчики, и API вернул массив объектов
        console.info(`🔄 Агрегация: Получен массив из ${data.length} объектов заказчиков.`);

        // Суммирование
        const totals = data.reduce((acc, item) => {
            acc.flight_count += item.flight_count || 0;
            acc.total_duration_min += item.total_duration_min || 0;
            return acc;
        }, { flight_count: 0, total_duration_min: 0 });

        finalMetricsData.flight_count = totals.flight_count;
        finalMetricsData.avg_duration_min = totals.flight_count > 0
                    ? parseFloat((totals.total_duration_min / totals.flight_count).toFixed(2))
                    : 0;

        // Сложные метрики берем из первого элемента, если они есть
        if (data.length > 0) {
            finalMetricsData = { ...finalMetricsData, ...data[0] };
        } else {
             // Если массив пуст
            finalMetricsData = {};
        }

    } else if (data && typeof data === 'object') {
        // Кейс: выбран конкретный заказчик, или API вернул один агрегированный объект
        finalMetricsData = data;
        console.info('✅ Агрегация: Получен один объект с агрегированными данными.');
    } else {
        // Кейс: пустой ответ, null или примитив
        console.warn('⚠️ Ответ от API не является объектом или массивом данных.');
        finalMetricsData = {};
    }

    // --- 6. ПРИСВОЕНИЕ МЕТРИК ---
    metrics.totalFlights = finalMetricsData.flight_count ?? null;
    if (finalMetricsData.avg_duration_min !== undefined && finalMetricsData.avg_duration_min !== null) {
      metrics.avgDuration = parseFloat(finalMetricsData.avg_duration_min.toFixed(2)); // Округление до 2 знаков
    } else {
      metrics.avgDuration = null;
    }
    metrics.growthPercent = finalMetricsData.growth_percent ?? null;
    metrics.peakLoad = finalMetricsData.peak_load ?? null;
    metrics.flightDensity = finalMetricsData.flight_density ?? null;
    metrics.zeroDays = finalMetricsData.zero_days ?? null;
    metrics.hourlyDistribution = finalMetricsData.hourly_distribution ?? null;

    growthPercentData.value = finalMetricsData.growth_data || [];
    missingMonths.value = finalMetricsData.missing_months || [];

    // --- 7. ФИНАЛЬНЫЙ ЛОГ ---
    console.log('🎯 Финальные метрики (обработаны):', {
      totalFlights: metrics.totalFlights,
      avgDuration: metrics.avgDuration,
      growthPercent: metrics.growthPercent,
      // ... (можно добавить другие финальные метрики)
    });
    console.groupEnd(); // Завершение группы лога

  } catch (error) {
    // --- 8. ОБРАБОТКА ОШИБОК И СБРОС ---
    const axiosError = error as any;

    // Сброс метрик при ошибке
    metrics.totalFlights = null;
    metrics.avgDuration = null;
    metrics.peakLoad = null;
    metrics.flightDensity = null;
    metrics.zeroDays = null;
    metrics.growthPercent = null;
    metrics.hourlyDistribution = null;
    growthPercentData.value = [];
    missingMonths.value = [];

    if (axiosError.response) {
      // Ошибка HTTP-ответа (статус 4xx или 5xx)
      const status = axiosError.response.status;
      console.error(`❌ ОШИБКА HTTP: Запрос к ${apiPath} завершился статусом ${status}.`);
      console.error('Детали ответа:', axiosError.response.data);
      console.error('Заголовки:', axiosError.response.headers);
    } else if (axiosError.request) {
      // Ошибка запроса (запрос отправлен, но нет ответа - таймаут, нет сети)
      console.error(`❌ ОШИБКА СЕТИ: Запрос к ${apiPath} не получил ответа.`);
      console.error('Детали запроса:', axiosError.request);
    } else {
      // Другие ошибки (ошибка настройки запроса)
      console.error('❌ КРИТИЧЕСКАЯ ОШИБКА: Ошибка настройки или выполнения запроса:', axiosError.message);
    }
    console.groupEnd(); // Завершение группы лога

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
    await axios.post('/api/upload', formData, {
      timeout: 300000, // 5 минут (300000 мс) по регламенту
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          const taskInArray = uploadTasks.value.find((t) => t.id === task.id);
          if (taskInArray) {
            taskInArray.progress = percentCompleted;
            // ---> ДОБАВЛЕНО: Изменяем статус на 'processing', когда загрузка завершена
            if (percentCompleted === 100) {
              taskInArray.status = 'processing';
            }
          }
        }
      },
    });
    const taskInArray = uploadTasks.value.find((t) => t.id === task.id);
    if (taskInArray) {
      taskInArray.status = 'success';
      handleFiltersUpdate(activeFilters.value);
    }
  }catch (error) {
    const taskInArray = uploadTasks.value.find((t) => t.id === task.id);
    if (taskInArray) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          taskInArray.status = 'timeout';
        } else {
          taskInArray.status = 'error';
          // <-- MODIFIED: Extract and store the error message from the API response
          taskInArray.errorMessage = error.response?.data?.error || 'Неизвестная ошибка сервера';
        }
      } else {
         taskInArray.status = 'error';
         taskInArray.errorMessage = 'Произошла непредвиденная ошибка';
      }
    }
    console.error('Ошибка при загрузке файла:', error);
  }
};

// --- НОВАЯ ФУНКЦИЯ: Обработка экспорта JSON ---
const handleExportJson = async () => {
  try {
    // Показываем индикатор загрузки (можно добавить глобальный loader, как в isLoadingMetrics)
    isLoadingMetrics.value = true; // Переиспользуем для блокировки UI

    const response = await axios.get('/api/report/export', {
      responseType: 'blob' // Для скачивания файла как blob
    });

    // Создаем URL для blob и скачиваем
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'full_report.json');
    document.body.appendChild(link);
    link.click();
    link.remove();

    // Успех: Можно показать toast или в UploadStatus добавить 'success' для экспорта
    console.log('JSON экспортирован успешно');
  } catch (error) {
    console.error('Ошибка экспорта JSON:', error);
    // Ошибка: Показать в UI (alert или модалка в стиле сайта)
    alert('Ошибка при экспорте JSON. Проверьте сервер или связь.');
  } finally {
    isLoadingMetrics.value = false;
  }
};


// --- ОБРАБОТЧИКИ СОБЫТИЙ ---
const handleFiltersUpdate = (filters: Filters) => {
  activeFilters.value = {
    from: filters.from || '2025-01-01',
    to: filters.to || '2025-12-31',
    metric: filters.metric || 'count',
    customer: filters.customer || null, // <-- ДОБАВЛЕНО
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
