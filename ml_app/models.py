from django.db import models

class BatteryData(models.Model):
    max_voltage_discharge = models.FloatField()
    min_voltage_charge = models.FloatField()
    rul = models.FloatField(blank=True, null=True)  # Predicted RUL

    def __str__(self):
        return f"Battery Data {self.id}"
        

