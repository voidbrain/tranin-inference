import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingLogs } from './training-logs';

describe('TrainingLogs', () => {
  let component: TrainingLogs;
  let fixture: ComponentFixture<TrainingLogs>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingLogs]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TrainingLogs);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
