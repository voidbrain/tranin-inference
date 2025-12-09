import { Routes } from '@angular/router';
import { Listen } from './listen/listen';
import { See } from './see/see';
import { TrainingLogs } from './training-logs/training-logs';
import { TrainingQueue } from './training-queue/training-queue';

export const routes: Routes = [
  { path: 'listen', component: Listen },
  { path: 'see', component: See },
  { path: 'training-logs', component: TrainingLogs },
  { path: 'training-queue', component: TrainingQueue },
  { path: '', redirectTo: '/listen', pathMatch: 'full' }
];
