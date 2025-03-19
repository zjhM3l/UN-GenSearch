import pandas as pd

data = [
    {"id": "001", "title": "Security Council Meeting 1001", "S/PV": "S/PV.1001", "Year": "2025", "Meeting_Number": "1001",
     "Day_Date_Time": "2025-01-15 10:00", "President": "John Doe", "Members": "USA, China, Russia",
     "agenda": "Nuclear Disarmament", "text": "The UN Security Council met today to discuss nuclear disarmament..."},
    
    {"id": "002", "title": "Security Council Meeting 1002", "S/PV": "S/PV.1002", "Year": "2025", "Meeting_Number": "1002",
     "Day_Date_Time": "2025-02-20 14:30", "President": "Jane Smith", "Members": "France, UK, Germany",
     "agenda": "Climate Change", "text": "Discussions focused on the impact of climate change on global security."}
]

df = pd.DataFrame(data)
df.to_csv("Mock_Meetings_Data.csv", index=False)

print("Mock data created successfully!")
