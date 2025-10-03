import { createRouter, createWebHistory } from 'vue-router';
import DashboardView from '../views/DashboardView.vue';
import LoginView from '../views/LoginView.vue';

const routes = [
  {
    path: '/',
    name: 'Login',
    component: LoginView
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: DashboardView
    // В будущем здесь можно добавить meta: { requiresAuth: true } для защиты роута
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;
