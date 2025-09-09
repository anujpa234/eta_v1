from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Employee APIs for LangGraph",
    description="Dummy Employee APIs for LangGraph Agent integration",
    version="1.0.0"
)

# Pydantic Models
class EmployeeInfo(BaseModel):
    employee_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    department: str
    role: str
    manager: str
    hire_date: str
    status: str
    location: str

class LeaveBalance(BaseModel):
    total: int
    used: int
    remaining: int

class EmployeeLeaveBalance(BaseModel):
    employee_id: str
    annual_leave: LeaveBalance
    sick_leave: LeaveBalance
    personal_leave: LeaveBalance
    last_updated: str

class Violation(BaseModel):
    id: str
    type: str
    severity: str
    date: str
    description: str
    resolved: bool

class EmployeeViolations(BaseModel):
    employee_id: str
    violations: List[Violation]
    total_violations: int
    last_violation: Optional[str]

class ErrorResponse(BaseModel):
    error: str
    employee_id: str
    message: str

# Dummy Database
EMPLOYEE_DATABASE = {
    "EMP001": {
        "info": EmployeeInfo(
            employee_id="EMP001",
            first_name="John",
            last_name="Doe",
            email="john.doe@company.com",
            phone="+1-555-0123",
            department="Engineering",
            role="Senior Software Engineer",
            manager="Alice Johnson",
            hire_date="2022-03-15",
            status="active",
            location="New York, NY"
        ),
        "leave_balance": EmployeeLeaveBalance(
            employee_id="EMP001",
            annual_leave=LeaveBalance(total=20, used=8, remaining=12),
            sick_leave=LeaveBalance(total=10, used=2, remaining=8),
            personal_leave=LeaveBalance(total=5, used=1, remaining=4),
            last_updated="2025-01-15"
        ),
        "violations": EmployeeViolations(
            employee_id="EMP001",
            violations=[
                Violation(
                    id="V001",
                    type="Late Arrival",
                    severity="low",
                    date="2024-11-15",
                    description="Arrived 30 minutes late without prior notice",
                    resolved=True
                )
            ],
            total_violations=1,
            last_violation="2024-11-15"
        )
    },
    "EMP002": {
        "info": EmployeeInfo(
            employee_id="EMP002",
            first_name="Sarah",
            last_name="Smith",
            email="sarah.smith@company.com",
            phone="+1-555-0456",
            department="Marketing",
            role="Marketing Manager",
            manager="Robert Chen",
            hire_date="2021-08-22",
            status="active",
            location="San Francisco, CA"
        ),
        "leave_balance": EmployeeLeaveBalance(
            employee_id="EMP002",
            annual_leave=LeaveBalance(total=22, used=15, remaining=7),
            sick_leave=LeaveBalance(total=12, used=0, remaining=12),
            personal_leave=LeaveBalance(total=5, used=3, remaining=2),
            last_updated="2025-01-15"
        ),
        "violations": EmployeeViolations(
            employee_id="EMP002",
            violations=[],
            total_violations=0,
            last_violation=None
        )
    },
    "EMP003": {
        "info": EmployeeInfo(
            employee_id="EMP003",
            first_name="Mike",
            last_name="Johnson",
            email="mike.johnson@company.com",
            phone="+1-555-0789",
            department="Sales",
            role="Sales Representative",
            manager="Lisa Davis",
            hire_date="2023-01-10",
            status="active",
            location="Chicago, IL"
        ),
        "leave_balance": EmployeeLeaveBalance(
            employee_id="EMP003",
            annual_leave=LeaveBalance(total=18, used=5, remaining=13),
            sick_leave=LeaveBalance(total=10, used=4, remaining=6),
            personal_leave=LeaveBalance(total=5, used=0, remaining=5),
            last_updated="2025-01-15"
        ),
        "violations": EmployeeViolations(
            employee_id="EMP003",
            violations=[
                Violation(
                    id="V002",
                    type="Missed Deadline",
                    severity="medium",
                    date="2024-12-05",
                    description="Failed to submit quarterly report on time",
                    resolved=False
                ),
                Violation(
                    id="V003",
                    type="Policy Violation",
                    severity="high",
                    date="2024-10-20",
                    description="Inappropriate use of company resources",
                    resolved=True
                )
            ],
            total_violations=2,
            last_violation="2024-12-05"
        )
    }
}

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Employee APIs for LangGraph Agent",
        "endpoints": [
            "/api/employee_info?employee_id={id}",
            "/api/leave_balance?employee_id={id}",
            "/api/employee_violations?employee_id={id}"
        ],
        "docs": "/docs"
    }

@app.get("/api/employee_info", response_model=EmployeeInfo)
async def get_employee_info(employee_id: str):
    """
    Get basic employee information including name, department, role, and contact details.
    """
    if employee_id not in EMPLOYEE_DATABASE:
        raise HTTPException(
            status_code=404, 
            detail=f"Employee with ID {employee_id} not found"
        )
    
    return EMPLOYEE_DATABASE[employee_id]["info"]

@app.get("/api/leave_balance", response_model=EmployeeLeaveBalance)
async def get_leave_balance(employee_id: str):
    """
    Get current leave balance including annual, sick, and personal leave days.
    """
    if employee_id not in EMPLOYEE_DATABASE:
        raise HTTPException(
            status_code=404, 
            detail=f"Employee with ID {employee_id} not found"
        )
    
    return EMPLOYEE_DATABASE[employee_id]["leave_balance"]

@app.get("/api/employee_violations", response_model=EmployeeViolations)
async def get_employee_violations(employee_id: str):
    """
    Get employee violation history including type, severity, and dates.
    """
    if employee_id not in EMPLOYEE_DATABASE:
        raise HTTPException(
            status_code=404, 
            detail=f"Employee with ID {employee_id} not found"
        )
    
    return EMPLOYEE_DATABASE[employee_id]["violations"]

# Bonus endpoint - get all employee data at once
@app.get("/api/employee_complete")
async def get_complete_employee_data(employee_id: str):
    """
    Get all employee data (info, leave balance, and violations) in one call.
    """
    if employee_id not in EMPLOYEE_DATABASE:
        raise HTTPException(
            status_code=404, 
            detail=f"Employee with ID {employee_id} not found"
        )
    
    employee_data = EMPLOYEE_DATABASE[employee_id]
    return {
        "employee_info": employee_data["info"],
        "leave_balance": employee_data["leave_balance"],
        "violations": employee_data["violations"]
    }

# List all available employees (helpful for testing)
@app.get("/api/employees")
async def list_employees():
    """
    Get list of all available employee IDs for testing.
    """
    return {
        "available_employees": list(EMPLOYEE_DATABASE.keys()),
        "total_count": len(EMPLOYEE_DATABASE)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)