# Generated by Django 3.0 on 2019-12-12 04:07

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app_core', '0002_resume_industry'),
    ]

    operations = [
        migrations.RenameField(
            model_name='resume',
            old_name='resume',
            new_name='pdf',
        ),
    ]
