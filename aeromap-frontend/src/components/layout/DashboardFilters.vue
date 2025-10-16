<template>
  <!--
    Панель фильтров: z-index: 100
  -->
  <div class="filters-panel">

    <!-- 1. ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМОВ -->
    <div class="mode-switcher" ref="switcherRef">
      <!-- Фон для активного состояния. -->
      <div
        class="mode-indicator"
        :style="indicatorStyle"
      ></div>

      <button
        class="mode-btn"
        ref="dateRangeBtnRef"
        :class="{ 'mode-active': filterMode === 'dateRange' }"
        @click="setFilterMode('dateRange')"
      >
        Date Range
      </button>
      <button
        class="mode-btn"
        ref="periodBtnRef"
        :class="{ 'mode-active': filterMode === 'period' }"
        @click="setFilterMode('period')"
      >
        Period
      </button>
    </div>

    <!-- 2. ГРУППА ФИЛЬТРОВ ПО РЕЖИМУ -->
    <div class="filter-controls-group">
      <!-- РЕЖИМ: ДИАПАЗОН ДАТ (Date Range) -->
      <div v-if="filterMode === 'dateRange'" class="date-range-inputs">

        <!-- From Date -->
        <div class="input-group">
          <label for="from-date" class="input-label">From Date</label>
          <div class="input-wrapper">
            <input
              id="from-date"
              type="date"
              v-model="filters.fromDate"
              class="date-input"
            />
            <div class="input-icon">
               <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clip-rule="evenodd" /></svg>
            </div>
          </div>
        </div>

        <!-- To Date -->
        <div class="input-group">
          <label for="to-date" class="input-label">To Date</label>
          <div class="input-wrapper">
            <input
              id="to-date"
              type="date"
              v-model="filters.toDate"
              class="date-input"
            />
            <div class="input-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clip-rule="evenodd" /></svg>
            </div>
          </div>
        </div>
      </div>

      <!-- РЕЖИМ: ПЕРИОД (Period: Year/Month) -->
      <div v-else class="period-selects">

        <!-- Выбор Года -->
        <div class="input-group">
          <label class="input-label">Год</label>
          <div class="dropdown-wrapper year-dropdown" @click="toggleDropdown('year')" :data-is-open="dropdowns.year">
            <button class="dropdown-button">
              <span class="font-medium">{{ filters.year }}</span>
              <svg class="dropdown-arrow" :class="{ 'rotate-180': dropdowns.year }" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
            </button>
            <div v-if="dropdowns.year" class="dropdown-menu">
              <button
                v-for="year in years"
                :key="year"
                class="dropdown-item"
                :class="{ 'item-active': filters.year === year }"
                @click.stop="selectPeriod('year', year)"
              >
                {{ year }}
                <svg v-if="filters.year === year" class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" /></svg>
              </button>
            </div>
          </div>
        </div>

        <!-- Выбор Месяца -->
        <div class="input-group">
          <label class="input-label">Месяц</label>
          <div class="dropdown-wrapper month-dropdown" @click="toggleDropdown('month')" :data-is-open="dropdowns.month">
            <button class="dropdown-button">
              <span class="font-medium">{{ filters.month ? getMonthName(filters.month) : 'Все' }}</span>
               <svg class="dropdown-arrow" :class="{ 'rotate-180': dropdowns.month }" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
            </button>
            <div v-if="dropdowns.month" class="dropdown-menu">
              <button
                class="dropdown-item"
                :class="{ 'item-active': filters.month === null }"
                @click.stop="selectPeriod('month', null)"
              >
                Все
                <svg v-if="filters.month === null" class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" /></svg>
              </button>
              <button
                v-for="month in months"
                :key="month.value"
                class="dropdown-item"
                :class="{ 'item-active': filters.month === month.value }"
                @click.stop="selectPeriod('month', month.value)"
              >
                {{ month.name }}
                <svg v-if="filters.month === month.value" class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" /></svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Кнопка Применить -->
    <button class="apply-btn" @click="applyFilters">Применить</button>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref, onMounted, watch, nextTick } from 'vue';

const emit = defineEmits(['filters-updated']);

// --- РЕФЫ ДЛЯ ДИНАМИЧЕСКОГО ПЕРЕКЛЮЧАТЕЛЯ ---
const switcherRef = ref<HTMLElement | null>(null);
const dateRangeBtnRef = ref<HTMLElement | null>(null);
const periodBtnRef = ref<HTMLElement | null>(null);
const indicatorStyle = ref({});


// --- СОСТОЯНИЕ ФИЛЬТРОВ И РЕЖИМОВ ---
const filterMode = ref<'dateRange' | 'period'>('dateRange'); // Default mode
const today = new Date().toISOString().split('T')[0];

const filters = reactive({
  // Режим Period
  year: '2025',
  month: null as string | null, // null = Все месяцы
  // Режим Date Range
  fromDate: '2025-01-01',
  toDate: today,
});

const dropdowns = reactive({
  year: false,
  month: false,
});

const currentYear = new Date().getFullYear();
const years = Array.from({ length: 5 }, (_, i) => String(currentYear + 2 - i)).filter(y => y >= '2023');
if (!years.includes('2025')) years.push('2025');
years.sort((a, b) => parseInt(b) - parseInt(a));


const months = [
  { value: '01', name: 'Январь' }, { value: '02', name: 'Февраль' },
  { value: '03', name: 'Март' }, { value: '04', name: 'Апрель' },
  { value: '05', name: 'Май' }, { value: '06', name: 'Июнь' },
  { value: '07', name: 'Июль' }, { value: '08', name: 'Август' },
  { value: '09', name: 'Сентябрь' }, { value: '10', name: 'Октябрь' },
  { value: '11', name: 'Ноябрь' }, { value: '12', name: 'Декабрь' },
];

const getMonthName = (monthValue: string | null) => {
    return months.find(m => m.value === monthValue)?.name || 'Все';
};

// --- ЛОГИКА ДИНАМИЧЕСКОГО ПЕРЕКЛЮЧАТЕЛЯ ---
const updateIndicatorStyle = () => {
  nextTick(() => {
    if (!switcherRef.value) return;

    const activeBtn = filterMode.value === 'dateRange'
      ? dateRangeBtnRef.value
      : periodBtnRef.value;

    if (activeBtn) {
      const switcherRect = switcherRef.value.getBoundingClientRect();
      const btnRect = activeBtn.getBoundingClientRect();

      indicatorStyle.value = {
        width: `${btnRect.width}px`,
        transform: `translateX(${btnRect.left - switcherRect.left}px)`,
      };
    }
  });
};

const setFilterMode = (mode: 'dateRange' | 'period') => {
  filterMode.value = mode;
  dropdowns.year = false;
  dropdowns.month = false;

  updateIndicatorStyle();
};



const toggleDropdown = (key: 'year' | 'month') => {
    if (key === 'year') {
        dropdowns.month = false;
    } else {
        dropdowns.year = false;
    }
    dropdowns[key] = !dropdowns[key];
};

const selectPeriod = (key: 'year' | 'month', value: string | null) => {
    filters[key] = value as any;
    toggleDropdown(key);
};

const applyFilters = () => {
  let fromDate: string;
  let toDate: string;

  if (filterMode.value === 'dateRange') {
    fromDate = filters.fromDate;
    toDate = filters.toDate;
  } else {
    const year = filters.year;
    const month = filters.month;

    if (month) {
      fromDate = `${year}-${month}-01`;
      const lastDayOfMonth = new Date(parseInt(year), parseInt(month), 0).getDate();
      toDate = `${year}-${month}-${String(lastDayOfMonth).padStart(2, '0')}`;
    } else {
      fromDate = `${year}-01-01`;
      toDate = `${year}-12-31`;
    }
  }

  emit('filters-updated', {
    from: fromDate,
    to: toDate,
    // Добавляем текущую метрику, чтобы не сбрасывать ее при смене дат
    metric: 'count' // или можно прокинуть из родителя
  });
};

// Наблюдатели
watch(filterMode, () => {
  updateIndicatorStyle();
  applyFilters();
});

watch(filters, applyFilters, { deep: true });

onMounted(() => {
    applyFilters();
    updateIndicatorStyle();
});

</script>

<style scoped>
/* ==================== 1. ОСНОВНОЙ КОНТЕЙНЕР ==================== */
.filters-panel {
  display: flex;
  align-items: center;
  gap: 24px;
  flex-wrap: wrap; /* Разрешаем перенос на новую строку */
  padding: 16px;
  background-color: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(50, 50, 50, 0.5);
  border-radius: 16px; /* Увеличили радиус */
  position: relative;
  z-index: 100;
}

/* ==================== 2. ПЕРЕКЛЮЧАТЕЛЬ РЕЖИМОВ ==================== */
.mode-switcher {
  display: inline-flex; /* Используем inline-flex для автоширины */
  align-items: center;
  background-color: rgba(0, 0, 0, 0.5);
  border-radius: 12px; /* Уменьшили радиус */
  padding: 4px;
  border: 1px solid rgba(50, 50, 50, 0.5);
  position: relative;
}

.mode-indicator {
    position: absolute;
    top: 4px;
    left: 0; /* Начальная позиция, будет изменена JS */
    height: calc(100% - 8px);
    background-color: #374151; /* bg-gray-700 */
    border-radius: 8px; /* Уменьшили радиус */
    transition: transform 0.3s ease, width 0.3s ease;
    z-index: 1;
}

.mode-btn {
  padding: 8px 16px;
  border-radius: 8px; /* Уменьшили радиус */
  font-size: 0.875rem;
  font-weight: 500;
  transition: color 0.3s ease;
  cursor: pointer;
  white-space: nowrap;
  border: none;
  background: transparent;
  color: #9ca3af; /* text-gray-400 */
  transition: color 0.3s ease;
  position: relative;
  z-index: 2;
}

.mode-btn:hover {
  color: #ffffff;
}

.mode-btn.mode-active {
  color: #fff;
}

/* ==================== 3. ГРУППА ФИЛЬТРОВ ==================== */
.filter-controls-group {
  display: flex;
  align-items: center;
  flex-wrap: wrap; /* Разрешаем перенос */
  gap: 16px;
}

.date-range-inputs, .period-selects {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

/* ==================== 4. СТИЛИ ДЛЯ INPUT/SELECT ==================== */
.input-group {
  display: flex;
  flex-direction: column;
  width: auto; /* Автоматическая ширина */
}

.input-label {
  color: #6b7280; /* text-gray-500 */
  font-size: 0.875rem;
  font-weight: 500;
  margin-bottom: 8px;
  white-space: nowrap;
}

.input-wrapper {
  position: relative;
}

.date-input, .dropdown-button {
  /* Базовые стили */
  background-color: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(55, 65, 81, 0.5); /* border-gray-800/50 */
  color: #fff;
  padding: 12px 16px;
  border-radius: 12px; /* rounded-xl */
  font-size: 0.95rem;
  transition: all 0.3s ease;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
  outline: none;

  /* Для выравнивания иконок и текста */
  display: inline-flex; /* Ключевое свойство для автоширины */
  align-items: center;
  white-space: nowrap;
}

.date-input {
  padding-right: 40px; /* Оставляем место для иконки */
}
.dropdown-button {
   justify-content: space-between;
   gap: 12px; /* Расстояние между текстом и стрелкой */
}

/* --- Новое свечение для Date Input --- */
.input-group:hover .date-input,
.date-input:focus {
    border-color: rgba(255, 255, 255, 0.7);
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.1);
}

/* --- Цветные эффекты свечения для дропдаунов --- */
.year-dropdown:hover .dropdown-button, .dropdown-wrapper[data-is-open="true"].year-dropdown .dropdown-button {
    border-color: #6d28d9; /* Purple */
    box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3);
}

.month-dropdown:hover .dropdown-button, .dropdown-wrapper[data-is-open="true"].month-dropdown .dropdown-button {
    border-color: #047857; /* Green */
    box-shadow: 0 4px 15px rgba(4, 120, 87, 0.3);
}


.input-icon {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  right: 16px;
  color: #6b7280; /* text-gray-500 */
  pointer-events: none;
  width: 1.1em;
  height: 1.1em;
}

/* ==================== 5. СТИЛИ ДЛЯ DROPDOWN ==================== */
.dropdown-wrapper {
    position: relative;
    cursor: pointer;
    width: auto;
}

.dropdown-arrow {
    width: 1.25em;
    height: 1.25em;
    color: #ffffff;
    transition: transform 0.3s ease;
}
.dropdown-arrow.rotate-180 {
    transform: rotate(180deg);
}

.dropdown-menu {
  position: absolute;
  top: calc(100% + 12px); /* Увеличили отступ */
  left: 0;
  background-color: #1a1a1a;
  border: 1px solid rgba(50, 50, 50, 0.8);
  border-radius: 12px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
  z-index: 101;
  min-width: 100%;
  overflow: hidden;
  max-height: 250px;
  overflow-y: auto;
  padding: 4px;
}
/* --- Стрелка для выпадающего меню --- */
.dropdown-menu::before {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 20px;
    width: 0;
    height: 0;
    border-left: 8px solid transparent;
    border-right: 8px solid transparent;
    border-bottom: 8px solid #323232; /* Цвет совпадает с border-color меню */
    z-index: 102;
}


.dropdown-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  text-align: left;
  padding: 8px 12px;
  color: #e0e0e0;
  background: transparent;
  border: none;
  cursor: pointer;
  transition: background-color 0.2s ease, color 0.2s ease;
  font-weight: 500;
  white-space: nowrap;
  border-radius: 8px; /* Скругляем элементы внутри */
}
.dropdown-item:hover {
  background-color: #374151; /* bg-gray-700 */
  color: #fff;
}
.dropdown-item.item-active {
    color: #fff;
    background-color: #4f46e5; /* bg-indigo-600, для контраста */
}
.checkmark {
    margin-left: 16px;
    color: #a7f3d0; /* Яркий, но не белый */
    width: 1.1em;
    height: 1.1em;
}

/* ==================== 6. КНОПКА "Применить" ==================== */
.apply-btn {
  margin-left: auto; /* Прижимаем вправо, если есть место */
  padding: 12px 24px;
  font-size: 0.95rem;
  font-weight: bold;
  color: #000;
  background: linear-gradient(to right, #ffc107, #ff9800);
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
  white-space: nowrap;
}
.apply-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(255, 193, 7, 0.4);
}

/* ==================== 7. АДАПТИВНОСТЬ ==================== */
@media (max-width: 768px) {
    .filters-panel {
        flex-direction: column;
        align-items: stretch;
    }
    .apply-btn {
        margin-left: 0;
        width: 100%;
    }
}
</style>

