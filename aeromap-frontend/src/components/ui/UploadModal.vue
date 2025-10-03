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
          <p>✓ Выбран файл: {{ selectedFile.name }}</p>
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
  const files = (event.target as HTMLInputElement).files;
  if (files && files.length > 0) {
    selectedFile.value = files[0];
  }
};

const handleDrop = (event: DragEvent) => {
  isDragActive.value = false;
  const files = event.dataTransfer?.files;
  if (files && files.length > 0 && files[0].name.endsWith('.xlsx')) {
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
/* Стили остаются такими же, как в моем предыдущем ответе */
.modal-overlay {
  position: fixed; inset: 0; background-color: rgba(0,0,0,0.7);
  backdrop-filter: blur(5px); display: flex; justify-content: center;
  align-items: center; z-index: 1000;
}
.modal-content {
  background: #0f2346; padding: 30px; border-radius: 12px;
  border: 1px solid #226bcb; width: 90%; max-width: 500px;
}
.modal-title { margin: 0 0 20px; color: #fff; }
.drop-zone {
  border: 2px dashed #30ceda; padding: 40px; text-align: center;
  border-radius: 8px; margin-bottom: 20px; color: #a0c3ff;
  transition: background-color 0.2s;
}
.drop-zone.is-active { background-color: rgba(48, 206, 218, 0.1); }
.browse-link { color: #30ceda; text-decoration: underline; cursor: pointer; }
.file-info { color: #64faff; }
.modal-actions { display: flex; justify-content: flex-end; gap: 15px; }
.btn {
  padding: 10px 20px; font-size: 0.9rem; font-weight: 500;
  border: none; border-radius: 8px; cursor: pointer; transition: all 0.2s ease;
}
.btn.primary { color: #fff; background-color: #30ceda; }
.btn.secondary { color: #a0c3ff; background-color: transparent; border: 1px solid #226bcb; }
</style>
