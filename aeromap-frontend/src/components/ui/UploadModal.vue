<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-content">
      <h3 class="modal-title">Загрузка нового отчёта</h3>
      <div
        class="drop-zone"
        :class="{ 'is-active': isDragActive }"
        @dragover.prevent="isDragActive = true"
        @dragleave.prevent="isDragActive = false"
        @drop.prevent="handleDrop"
      >
        <input type="file" ref="fileInput" @change="handleFileSelect" accept=".xlsx" hidden />
        <p v-if="!selectedFile">Перетащите XLSX файл сюда или <span class="browse-link" @click="openFileBrowser">выберите файл</span></p>
        <div v-else class="file-info">
          <p>✓ Выбран файл: **{{ selectedFile.name }}**</p>
        </div>
      </div>
      <div class="modal-actions">
        <button class="btn secondary" @click="$emit('close')">Отмена</button>
        <button class="btn primary" @click="startUpload" :disabled="!selectedFile">
          Начать загрузку
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const emit = defineEmits(['start-upload', 'close']);
const fileInput = ref<HTMLInputElement | null>(null);
const selectedFile = ref<File | null>(null);
const isDragActive = ref(false);

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const files = target.files;
  if (files && files.length > 0) {
    selectedFile.value = files[0];
  }
};

const handleDrop = (event: DragEvent) => {
  isDragActive.value = false;
  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    // Принимаем только первый файл
    selectedFile.value = files[0];
  }
};

const openFileBrowser = () => fileInput.value?.click();

const startUpload = () => {
  if (!selectedFile.value) return;
  // Отправляем событие наверх, передавая сам файл
  emit('start-upload', selectedFile.value);
  // Сразу закрываем окно
  emit('close');
};
</script>

<style scoped>
.modal-overlay {
  position: fixed; inset: 0; background-color: rgba(0,0,0,0.7);
  backdrop-filter: blur(5px); display: flex; justify-content: center;
  align-items: center; z-index: 1000;
}
.modal-content {
  /* Фон: Темно-серый */
  background: #111111;
  padding: 30px; border-radius: 12px;
  /* Граница: Темно-серая */
  border: 1px solid #333333;
  width: 90%; max-width: 500px;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.7);
}
.modal-title { margin: 0 0 20px; color: #fff; }
.drop-zone {
  /* Рамка Drag & Drop: Янтарный пунктир */
  border: 2px dashed #ffc107;
  padding: 40px; text-align: center;
  border-radius: 8px; margin-bottom: 20px;
  /* Цвет текста: Серый */
  color: #aaaaaa;
  transition: background-color 0.2s;
}
.drop-zone.is-active {
  /* Активный фон: Прозрачный янтарный */
  background-color: rgba(255, 193, 7, 0.1);
}
.browse-link {
  /* Ссылка выбора файла: Янтарный */
  color: #ffc107;
  text-decoration: underline;
  cursor: pointer;
}
.file-info {
  color: #e0e0e0;
}
.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
/* Стили кнопок из AppHeader */
.btn {
  padding: 10px 20px;
  font-size: 0.9rem;
  font-weight: 500;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.btn.primary {
  color: #000000;
  background-color: #ffc107;
}
.btn.primary:hover:not(:disabled) {
  background-color: #ffac30;
}
.btn.primary:disabled {
  background-color: #333333;
  color: #666666;
  cursor: not-allowed;
}
.btn.secondary {
  color: #e0e0e0;
  background-color: transparent;
  border: 1px solid #333333;
}
.btn.secondary:hover {
  background-color: #1a1a1a;
}
</style>
