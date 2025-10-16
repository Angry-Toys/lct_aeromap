<template>
  <div v-if="visible" class="modal-overlay" @click.self="clearFocus">
    <div ref="modalContentRef" class="modal-content">
      <h3 class="modal-title">Экспорт отчёта</h3>
      <p class="modal-subtitle">Выберите элементы для включения в PNG файл</p>

      <div class="selection-container">
        <label
          v-for="chart in charts"
          :key="chart.id"
          class="checkbox-label"
          :class="{ 'is-focused': focusedId === chart.id }"
          @click="setFocus(chart.id)"
        >
          <input type="checkbox" v-model="selectedCharts" :value="chart.id" />
          <span class="custom-checkbox">
            <svg class="checkmark-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.052-.143z" clip-rule="evenodd" />
            </svg>
          </span>
          <span class="checkbox-text">{{ chart.name }}</span>
        </label>
      </div>

      <div class="modal-actions">
        <button class="btn cancel-btn" @click="close">Отмена</button>
        <button class="btn export-btn" @click="generateReport">Сгенерировать</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue';

const props = defineProps<{ visible: boolean }>();
const emit = defineEmits(['close', 'generate']);

// --- Логика для сохранения состояния фокуса ---
const focusedId = ref<string | null>(null); // Хранит ID элемента, который должен быть подсвечен
const modalContentRef = ref<HTMLElement | null>(null);

const setFocus = (id: string) => {
  focusedId.value = id;
};

const clearFocus = () => {
  focusedId.value = null;
};

// --- Обработчик клика вне модального окна для сброса фокуса ---
const handleClickOutside = (event: MouseEvent) => {
  if (modalContentRef.value && !modalContentRef.value.contains(event.target as Node)) {
    clearFocus();
  }
};

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
});

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside);
});

// Сбрасываем фокус при закрытии модалки
watch(() => props.visible, (newValue) => {
  if (!newValue) {
    clearFocus();
  }
});

// --- Остальная логика компонента ---
const charts = ref([
  { id: 'metrics', name: 'Сводные метрики' },
  { id: 'map', name: 'Карта регионов' },
  { id: 'topRegions', name: 'Топ регионов по активности' },
  { id: 'hourlyActivity', name: 'График почасовой активности' },
]);

const selectedCharts = ref(charts.value.map(c => c.id));

const close = () => emit('close');
const generateReport = () => {
  if (selectedCharts.value.length === 0) {
    alert('Пожалуйста, выберите хотя бы один элемент для экспорта.');
    return;
  }
  emit('generate', selectedCharts.value);
  close();
};
</script>

<style scoped>
/* --- 1. ИСПРАВЛЕННЫЕ СТИЛИ ОКНА --- */
.modal-overlay {
  position: fixed;
  inset: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  backdrop-filter: blur(8px);
  background-color: rgba(0, 0, 0, 0.7); /* Добавим легкое затемнение */
}

.modal-content {
  background-color: rgba(17, 17, 17, 0.8);
  border: 1px solid rgba(50, 50, 50, 0.5);
  border-radius: 16px;
  width: 500px;
  padding: 32px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  overflow-y: auto;
  animation: fadeInScale 0.1s ease-out forwards;
}

.modal-title, .modal-subtitle, .modal-actions {
  /* Стили скопированы из предыдущей рабочей версии */
  text-align: center;
}
.modal-title { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0 0 8px; }
.modal-subtitle { color: #aaaaaa; margin: 0 0 24px; font-size: 0.9rem; }
.modal-actions { margin-top: 24px; display: flex; justify-content: flex-end; gap: 12px; }

/* --- 2. ИСПРАВЛЕННЫЕ СТИЛИ КНОПОК --- */
.btn {   padding: 12px 24px;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
}
.cancel-btn {
  color: #e0e0e0;
  background-color: #2a2a2a;
}
.cancel-btn:hover {   background-color: #333; }
.export-btn { background: linear-gradient(to right, #6d28d9, #4f46e5); color: white; box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3); }
.export-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(109, 40, 217, 0.4); }


/* --- 3. ЧЕКБОКСЫ С ПРАВИЛЬНОЙ МЕХАНИКОЙ --- */
.selection-container { display: flex; flex-direction: column; gap: 10px; margin-bottom: 32px; }
.checkbox-label { display: flex; align-items: center; cursor: pointer; padding: 10px 12px; border-radius: 8px; }
.checkbox-label input[type="checkbox"] { display: none; }

.custom-checkbox {
  width: 1.1rem;
  height: 1.1rem;
  /* Резервируем место, чтобы избежать "дергания" */
  border: 2px solid transparent;
  /* Имитируем тонкую белую границу */
  box-shadow: inset 0 0 0 1px #8c8c8c;
  border-radius: 4px;
  margin-right: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all 0.1s linear;
}

.checkmark-icon { width: 0.8rem; height: 0.8rem; color: white; opacity: 0; transform: scale(0.5); transition: all 0.15s ease-out; }

/* --- ЭТАП 1: МГНОВЕННАЯ БЕЛАЯ ГРАНИЦА (при зажатой мыши) --- */
.checkbox-label:active .custom-checkbox {
  transition-duration: 0s;
  box-shadow: none;
  border-color: white;
}

/* --- ЭТАП 2: ПОСТОЯННАЯ ФИОЛЕТОВАЯ ПОДСВЕТКА (когда элемент "в фокусе") --- */
.checkbox-label.is-focused .custom-checkbox {
  box-shadow: 0 0 0 3px #6d28d9;
  border-color: #ffffff;
}

/* Стили ВЫБРАННОГО чекбокса */
.checkbox-label input[type="checkbox"]:checked + .custom-checkbox {
  background-color: #6d28d9;
  border-color: #6d28d9;
  box-shadow: none;
}
.checkbox-label input[type="checkbox"]:checked + .custom-checkbox .checkmark-icon { opacity: 1; transform: scale(1); }

/* Комбинация: ВЫБРАННЫЙ и В ФОКУСЕ */
.checkbox-label.is-focused input[type="checkbox"]:checked + .custom-checkbox {
  box-shadow: 0 0 0 3px #6d28d9;
  border-color: #ffffff;
}

.checkbox-text { color: #d1d5db; font-size: 0.95rem; font-weight: 500; }

@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

</style>
