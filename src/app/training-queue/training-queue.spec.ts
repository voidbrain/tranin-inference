import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingQueue } from './training-queue';

describe('TrainingQueue', () => {
  let component: TrainingQueue;
  let fixture: ComponentFixture<TrainingQueue>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingQueue]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TrainingQueue);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
