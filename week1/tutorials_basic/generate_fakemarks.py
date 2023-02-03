#-----------------------------------------------------------------------
# I got our email list for this course, and I'll take a toy exam
# and generate marks for students in their maths, bio, phy and
# chemistry exam using random number generator using python.
#
# Since there is less no of people in this email list, I'll generate
# 200 students name by mixing our first name and last name randomly
#  
# So, at the end I should produce a text file with columns
# first name, last name, maths, phy, chem, bio
#-------------------------------------------------------------------------

# import libraries
import numpy as np
import random


## gather data

bad_data= "Deep Mazumdar <deepkamal.mazumdar@students.iiserpune.ac.in>,\
Siddhant Sen <siddhant.sen@students.iiserpune.ac.in>,Eksha <chaudhary.eksharani@students.iiserpune.ac.in>,\
Ojasvi Sharma <ojasvi.sharma@students.iiserpune.ac.in>,Om Vandra <om.vandra@students.iiserpune.ac.in>,\
Asif Mohammed L <asif.mohammed@students.iiserpune.ac.in>,Siddharth Seetharaman <siddharth.seetharaman@students.iiserpune.ac.in>,\
Pankaj Gupta <pankaj.gupta@students.iiserpune.ac.in>,Harikrishnan KB <harikrishnan.kb@students.iiserpune.ac.in>,\
joel.sunil@students.iiserpune.ac.in,shuvarati.roy@students.iiserpune.ac.in,\
Khadekar Pavan <khadekar.pavan@students.iiserpune.ac.in>, Avinash Mahapatra <avinash.mahapatra@students.iiserpune.ac.in>,\
Nisarg Vyas <nisarg.vyas@students.iiserpune.ac.in>,jbharathi.kannan@students.iiserpune.ac.in,\
Katha Ganguly <katha.ganguly@students.iiserpune.ac.in>,Rajeev Ranjan <rajeev.ranjan@students.iiserpune.ac.in>,\
Pushkar Saoji <pushkar.saoji@students.iiserpune.ac.in>,arindam.ghara@students.iiserpune.ac.in,\
ABHINAV DHAWAN <abhinav.dhawan@students.iiserpune.ac.in>,INDRAKANTY SURYA <indrakanty.surya@students.iiserpune.ac.in>,\
SHASHANK SONI <shashank.soni@students.iiserpune.ac.in>,Varna Shenoy <varna.shenoy@students.iiserpune.ac.in>,\
shreyas.nadiger@students.iiserpune.ac.in,ANURAKTI GUPTA <anurakti.gupta@students.iiserpune.ac.in>,\
Pranav Maheshwari <pranav.maheshwari@students.iiserpune.ac.in>,SOHAM CHANDAK <soham.chandak@students.iiserpune.ac.in>,\
Mittal Shree <mittal.shree@students.iiserpune.ac.in>,rayees.as@students.iiserpune.ac.in,\
Vikhyat Sharma <vikhyat.sharma@students.iiserpune.ac.in>,Arnab Laha <laha.arnab@students.iiserpune.ac.in>,\
Prachurjya Hazarika <prachurjya.hazarika@students.iiserpune.ac.in>,Kumar Yash <kumar.yash@students.iiserpune.ac.in>,\
Anantha S Rao <anantha.rao@students.iiserpune.ac.in>,Arun Ravi <arun.ravi@students.iiserpune.ac.in>"



# filter out names out of this string
names=[item.split("<")[0] for item in (bad_data.split(","))]

#clean the data and create two list of student's first name and student's last name
names = [item.rstrip('@students.iiserpune.ac.in') for item in names]
names = [item.rstrip(' ') for item in names]
names = [item.lstrip(' ') for item in names]

first_names = [item.split(' ')[0] for item in names if ' ' in item]
last_names  = [item.split(' ')[1] for item in names if ' ' in item]


# define a function which should generate fake marks in a toy exam
def generate_fakemarks(size):  # size = no of students
    group=[]
    for i in range(size):
        group.append(random.choice(first_names) +" "+ random.choice(last_names))
    #Generate fake marks
    maths = np.random.randint(0,100,size)
    phy   = np.random.randint(maths/8,100,size)      ## creating bias in exam
    chem  = np.random.randint(maths/4,100,size)        ## creating bias in exam
    bio   = np.random.randint(maths/5,100,size)       ## creating bias in exam
    result= zip(group,maths,phy,chem,bio)
    
    return list(result)


## Take the toy exam of 200 students
result= generate_fakemarks(200)

## create a text file with the outcome of this exam
file= open("fake_exam.txt", "w")
#print(list(result))
for name,marksA,marksB,marksC,marksD in list(result):
    #print(name.strip(' '))
    file.write(name.strip(' ')+" ")
    file.write(str(int(marksA))+" ")
    file.write(str(marksB)+" ")
    file.write(str(marksC)+" ")
    file.write(str(marksD)+" ")
    file.write("\n")
file.close()    


print("\n\n You have taken a fake exam !!!!")
print(" Do a data analysis now using pandas dataframe!\n\n")



