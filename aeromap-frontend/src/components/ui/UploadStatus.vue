<template>
  <div class="upload-status-container" v-if="tasks.length > 0">
    <div v-for="task in tasks" :key="task.id" class="upload-task">
      <div class="task-info">
        <span class="filename">{{ task.file.name }}</span>
        <span class="status" :class="task.status">{{ getStatusText(task.status, task.progress) }}</span>
      </div>
      <div class="progress-bar-wrapper">
        <div
          class="progress-bar"
          :class="task.status"
          :style="{ width: task.progress + '%' }"
        ></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { PropType } from 'vue';

// Определяем тип для задачи загрузки
interface UploadTask {
  id: number;
  file: File;
  progress: number;
  status: 'uploading' | 'success' | 'error' | 'timeout';
}

defineProps({
  tasks: {
    type: Array as PropType<UploadTask[]>,
    required: true
  }
});

// Функция для отображения текста статуса
const getStatusText = (status: UploadTask['status'], progress: number) => {
  switch (status) {
    case 'uploading':
      return `Загрузка... ${progress}%`;
    case 'success':
      return '✓ Успешно';
    case 'error':
      return '✗ Ошибка';
    case 'timeout':
      return '✗ Время ожидания истекло';
    default:
      return '';
  }
};
</script>

<style scoped>
.upload-status-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 350px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  z-index: 2000;
}
.upload-task {
  background-color: #0f2346;
  border: 1px solid #226bcb;
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.task-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: #fff;
}
.filename {
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
}
.status {
  font-size: 0.8rem;
  font-weight: bold;
}
.status.uploading { color: #a0c3ff; }
.status.success { color: #64faff; }
.status.error, .status.timeout { color: #ff8e8e; }

.progress-bar-wrapper {
  width: 100%;
  height: 6px;
  background-color: #0a1929;
  border-radius: 3px;
  overflow: hidden;
}
.progress-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}
.progress-bar.uploading { background-color: #30ceda; }
.progress-bar.success { background-color: #64faff; }
.progress-bar.error, .progress-bar.timeout { background-color: #ff8e8e; }
</style>
