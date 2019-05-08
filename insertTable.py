   
#Create HTML Table
def insertTable(names,last,email,city):
    
    strTable = "{% extends 'examples/credit_risk.html' %}{% block insertTable %}<table class='table table-dark'><tr><th>Full Name</th><th>email</th><th>City</th></tr>"
    
    for i in range(len(names)):
        full_name = names[i]+' '+ last[i]
        email = email[i]
        city= city[i]
        strRW = "<tr><td>"+full_name+ "</td><td>"+email+"</td><td>"+city+"</td></tr>"
        strTable = strTable+strRW
        
    strTable = strTable+"</table>{% endblock %}"
    
    hs = open("templates\examples\dataTable.html", 'w')
    hs.write(strTable)
    
    print (strTable)