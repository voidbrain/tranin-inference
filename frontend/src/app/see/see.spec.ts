import { ComponentFixture, TestBed } from '@angular/core/testing';

import { See } from './see';

describe('See', () => {
  let component: See;
  let fixture: ComponentFixture<See>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [See]
    })
    .compileComponents();

    fixture = TestBed.createComponent(See);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
