<template>
  <div class="metric-card">
    <div v-if="isLoading" class="loader">...</div>
    <template v-else>
      <p class="title">{{ title }}</p>
      <h3 class="value">{{ value ?? '-' }} <span v-if="unit">{{ unit }}</span></h3>
    </template>
  </div>
</template>
<script setup lang="ts">
defineProps<{
  title: string;
  value: number | null;
  isLoading: boolean;
  unit?: string;
}>();
</script>
<style scoped>
.metric-card {
  background: #0f2346;
  border: 1px solid #226bcb;
  border-radius: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.title {
  margin: 0 0 8px;
  color: #a0c3ff;
  font-size: 1rem;
}
.value {
  margin: 0;
  font-size: 2.2rem;
  font-weight: 700;
  color: #fff;
}
.unit {
  font-size: 1rem;
  font-weight: 400;
  color: #a0c3ff;
  margin-left: 4px;
}
.loader-skeleton .skeleton-line {
  background: #1a325e;
  border-radius: 4px;
}
.loader-skeleton .title {
  width: 70%;
  height: 16px;
  margin-bottom: 12px;
}
.loader-skeleton .value {
  width: 50%;
  height: 32px;
}
</style>
