# Generated by Django 3.0 on 2019-12-12 02:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app_core', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='resume',
            name='industry',
            field=models.CharField(blank=True, max_length=50),
        ),
    ]
