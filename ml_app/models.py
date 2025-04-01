from django.db import models

class BatteryData(models.Model):
    cycle_index = models.FloatField()
    discharge_time = models.FloatField()
    decrement_36_34V = models.FloatField()
    max_voltage_discharge = models.FloatField()
    min_voltage_charge = models.FloatField()
    time_at_415V = models.FloatField()
    time_constant_current = models.FloatField()
    charging_time = models.FloatField()
    rul = models.FloatField(blank=True, null=True)  # Predicted RUL

    def __str__(self):
        return f"Battery Data {self.id}"
