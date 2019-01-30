plt.figure(figsize=(12,12))

# Plot the feature "Children"
plt.subplot(3,4,1)
plt.scatter(train.iloc[:,1],train.iloc[:,-1])
plt.xticks([], [])
plt.yticks([],[])
plt.title('Number of Children')

# Plot the extra features
for i in range(2,13,1):
    plt.subplot(3,4,i)
    plt.scatter(train.iloc[:,-i],train.iloc[:,-1])
    title=["", "", "Sqrt of Children Feature","Has Link?", "Has Quote?", "Has Reference?", "Has Exclamation Mark?", "Number of Words","Punctuation/Word", "Average Letteres/Word", "Is There A Question?", "Swear Words Count", "Misspelled Words Count"]
    plt.title(title[i])
    plt.xticks([], [])
    plt.yticks([],[])
    
# Save the image 
plt.savefig('features.png')