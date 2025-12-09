import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Listen } from './listen';

describe('Listen', () => {
  let component: Listen;
  let fixture: ComponentFixture<Listen>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Listen]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Listen);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
