<template>
  <div class="metric-card" @click="$emit('card-click')">
    <div v-if="isLoading" class="loader-skeleton">
      <div class="skeleton-line title-line"></div>
      <div class="skeleton-line value-line"></div>
      <div class="skeleton-line progress-line"></div>
    </div>

    <template v-else>
      <div class="top-content">
        <div class="icon-wrapper">
          <slot name="icon">
            <i class="fas fa-chart-line"></i>
          </slot>
        </div>

        <div class="value-group">
          <p class="value">{{ value ?? '-' }} <span v-if="unit" class="unit-inline">{{ unit }}</span></p>
          <p class="title">{{ title }}</p>
        </div>
      </div>

      <div v-if="progressValue !== undefined" class="progress-bar-wrapper">
        <div class="progress-bar" :style="{ width: progressValue + '%' }"></div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
// Добавим новый пропс для прогресс-бара.
// ProgressValue должен быть числом от 0 до 100
defineProps<{
  title: string;
  value: number | string | null; // Разрешаем строку для форматирования
  isLoading: boolean;
  unit?: string;
  progressValue?: number;
}>();

defineEmits(['card-click']);
</script>

<style scoped>
/* ======================== 1. CARD BASE ======================== */
.metric-card {
  /* Уменьшаем вертикальный padding с 20px до 16px */
  padding: 16px;

  background: rgba(0, 0, 0, 0.4);
  border: 1px solid #333333;
  border-radius: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.metric-card:hover {
  border-color: #ffc107;
  box-shadow: 0 0 10px rgba(255, 193, 7, 0.2);
}

/* ======================== 2. TOP CONTENT (Icon + Text) ======================== */
.top-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  /* Уменьшаем отступ между верхним блоком и нижним (прогресс-баром) */
  /* margin-bottom: 10px; */
}

/* ======================== 3. ICON WRAPPER (без изменений) ======================== */
.icon-wrapper {
  width: 40px;
  height: 40px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, rgba(255, 193, 7, 0.5) 0%, rgba(180, 100, 0, 0.5) 100%);
  color: #fff;
  font-size: 1.2rem;
}

/* ======================== 4. VALUES ======================== */
.value-group {
  text-align: right;
}

.value {
  /* Уменьшаем font-size */
  font-size: 1.6rem;
  font-weight: 700;
  color: #ffc107;
  /* Уменьшаем отступ под значением */
  margin: 0 0 2px;
  line-height: 1.1;
}

.unit-inline {
  font-size: 0.8rem;
  font-weight: 400;
  color: #aaaaaa;
}

.title {
  font-size: 0.8rem;
  color: #aaaaaa;
  margin: 0;
  white-space: nowrap;
}

/* ======================== 5. PROGRESS BAR ======================== */
.progress-bar-wrapper {
  height: 4px;
  background: #1a1a1a;
  border-radius: 4px;
  overflow: hidden;
  /* Уменьшаем отступ сверху */
  margin-top: 8px;
}

.progress-bar {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, #ffac30 0%, #ffc107 100%);
  transition: width 0.5s ease;
}

/* ======================== 6. SKELETON LOADER (корректировка под новые размеры) ======================== */
.loader-skeleton {
  padding: 4px 0;
}
.skeleton-line {
  background: #1a1a1a;
  border-radius: 4px;
  animation: loading-pulse 1.5s infinite ease-in-out;
}
.title-line {
  height: 10px;
  width: 60%;
  margin-bottom: 6px; /* Корректируем отступ */
  margin-left: auto;
}
.value-line {
  height: 20px; /* Корректируем высоту */
  width: 80%;
  margin-bottom: 8px; /* Корректируем отступ */
  margin-left: auto;
}
.progress-line {
    height: 4px;
    width: 100%;
    margin-top: 6px; /* Корректируем отступ */
}
@keyframes loading-pulse {
  0% { opacity: 0.6; }
  50% { opacity: 1; }
  100% { opacity: 0.6; }
}

</style>
