import { Routes } from '@angular/router';
import { Listen } from './listen/listen';
import { See } from './see/see';

export const routes: Routes = [
  { path: 'listen', component: Listen },
  { path: 'see', component: See },
  { path: '', redirectTo: '/listen', pathMatch: 'full' }
];
