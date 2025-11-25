import os
import tempfile
import subprocess
import smtplib

import xml.etree.ElementTree as ET

from email.mime.text import MIMEText
from .raml.main import SmaliMalwareAnalyzer
from ..celery import schedule_task
from .raml.report_generator import ReportGenerator
from .. import models
from ..database import get_db

@schedule_task
async def analyze_apk_with_raml(apk_path: str, user_email: str, apk_report_id: int) -> dict:
    smali_output = tempfile.mkdtemp(prefix="smali_code")
    decode_cmd = [
        "apktool",
        "decode",
        apk_path,
        "-o",
        smali_output,
        "-f"
    ]

    try:
        subprocess.run(decode_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        package_name = get_package_name(smali_output)
        analyzer = SmaliMalwareAnalyzer(smali_folder=smali_output, package_name=package_name)
        await analyzer.setup_system(force_rebuild=True)
        result = await analyzer.analyze_behaviors(list(range(1, 13)))
        report_generator = ReportGenerator(output_dir="reports")
        tempfile_path = "temp_report.md"
        report_generator.save_summary_report(result, tempfile_path)
        with open(os.path.join("reports", tempfile_path), "r") as f:
            markdown_content = f.read()

        db = next(get_db())
        apk_report = db.query(models.APKReport).filter(models.APKReport.id == apk_report_id).first()
        apk_report.markdown_report = markdown_content
        db.commit()

        send_email(user_email, "Your request is complete. Jump to the dashboard to view it!")
        return result

    except subprocess.CalledProcessError as e:
        send_email(user_email, "Bad Luck! Your request failed")
        print(f"Command failed with exit code {e.returncode}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    
    except Exception as e:
        send_email(user_email, "Bad Luck! Your request failed")
        pass


def get_package_name(apktool_output_dir):
    """Extract package name from AndroidManifest.xml"""
    manifest_path = os.path.join(apktool_output_dir, "AndroidManifest.xml")
    tree = ET.parse(manifest_path)
    root = tree.getroot()
    return root.attrib['package']


def send_email(recipient: str, body: str):
    sender = os.environ.get('EMAIL_ADDRESS')
    password = os.environ.get('EMAIL_PASSWORD')
    msg = MIMEText(body)
    msg['Subject'] = "RAML Request Status Update"
    msg['From'] = sender # type: ignore
    msg['To'] = recipient
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password) # type: ignore
       smtp_server.sendmail(sender, recipient, msg.as_string()) # type: ignore
